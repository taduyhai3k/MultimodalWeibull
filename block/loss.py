import torch
import torch.nn as nn
import torch.nn.functional as f

class CLUBStable(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_size: int | None = None,
    ):
        super().__init__()

        if hidden_size is None:
            self.p_mu = nn.Linear(x_dim, y_dim)
        else:
            self.p_mu = nn.Sequential(
                nn.Linear(x_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, y_dim),
            )

    def get_distribution(self, x):
        mu = self.p_mu(x)
        return mu

    def forward(self, x, y):
        """
        x: [B, x_dim]
        y: [B, y_dim]
        """
        mu = self.get_distribution(x)  # [B, y_dim]

        # Cặp đúng (x_i, y_i)
        positive_dist = 0.5 * (
            mu - y
        ).square().sum(dim=-1)  # [B]

        # Tất cả cặp (x_i, y_j)
        pairwise_dist = 0.5 * (
            mu[:, None, :] - y[None, :, :]
        ).square().sum(dim=-1)  # [B, B]

        negative_dist = pairwise_dist.mean(dim=1)  # [B]

        mi_upper_bound = (
            negative_dist - positive_dist
        ).mean()

        return mi_upper_bound
    def loss_loglikelihood(self, x, y):
        mu = self.get_distribution(x)  # [B, y_dim]
        log_likelihood = -0.5 * (
            mu - y
        ).square().sum(dim=-1)  # [B]
        return -1 * log_likelihood.mean()
    

class WeibullLossandInference(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super(WeibullLossandInference, self).__init__()
        self.eps = eps


    def forward(self,x: torch.Tensor,
        times: torch.Tensor,
        uncensored: torch.Tensor,
        shapes: torch.Tensor,
        scales: torch.Tensor,
        weights: torch.Tensor,
        reduction: str = "mean",
    ):
        """
        Negative log-likelihood cho Weibull Mixture có right-censoring.

        Quy ước Weibull:
            f(t) = lambda * k * t^(k - 1) * exp(-lambda * t^k)
            S(t) = exp(-lambda * t^k)

        Quy ước censor:
            uncensored = 1: mẫu không bị censor
            uncensored = 0: mẫu bị censor

        Parameters
        ----------
        x:
            Tensor đầu vào, có thể là các đặc trưng bổ sung.
            Shape: [B, D]
        times:
            Thời gian xảy ra sự kiện hoặc thời điểm censor.
            Shape: [B] hoặc [B, 1]

        uncensored:
            Chỉ báo uncensored.
            Shape: [B] hoặc [B, 1]

        shapes:
            Tham số shape k của từng thành phần.
            Shape: [B, M, D]

        scales:
            Tham số scale của từng thành phần.
            Shape: [B, M, D]

        weights:
            Trọng số mixture đã đi qua softmax.
            Shape: [B, M]

        reduction:
            "mean", "sum" hoặc "none".

        Returns
        -------
        loss:
            Scalar nếu reduction là "mean" hoặc "sum".
            Tensor [B] nếu reduction là "none".
        """

        shape_logits = torch.einsum(
                "bd,bmd->bm",
                x,
                shapes
            )

        scale_logits = torch.einsum(
                "bd,bmd->bm",
                x,
                scales
            )

        shapes = f.softplus(shape_logits) + self.eps
        scales = f.softplus(scale_logits) + self.eps


        # [B, 1] -> [B]
        times = times.reshape(-1)
        uncensored = uncensored.reshape(-1)

        # Bảo đảm các giá trị hợp lệ
        times = times.clamp_min(self.eps)
        shapes = shapes.clamp_min(self.eps)
        scales = scales.clamp_min(self.eps)
        weights = weights.clamp_min(self.eps)

        #Chuẩn hóa lại để tổng trọng số bằng 1
        weights = weights / weights.sum(
            dim=-1,
            keepdim=True
        )

        # [B] -> [B, 1], tự động broadcast với [B, M]
        log_times = torch.log(times).unsqueeze(-1)

        log_shapes = torch.log(shapes)
        log_scales = torch.log(scales)

        # log(lambda * t^k) = log(lambda) + k * log(t)
        log_cumulative_hazard = (
            log_scales
            + shapes * log_times
        )

        # Tránh exp() overflow trong float32
        max_log_value = 80.0

        cumulative_hazard = torch.exp(
            log_cumulative_hazard.clamp(max=max_log_value)
        )

        # log S_m(t) = -lambda_m * t^k_m
        log_survival_components = -cumulative_hazard

        # log f_m(t)
        log_pdf_components = (
            log_scales
            + log_shapes
            + (shapes - 1.0) * log_times
            - cumulative_hazard
        )

        # log(pi_m)
        log_weights = torch.log(weights)

        # log sum_m[pi_m * f_m(t)]
        log_pdf_mixture = torch.logsumexp(
            log_weights + log_pdf_components,
            dim=-1
        )

        # log sum_m[pi_m * S_m(t)]
        log_survival_mixture = torch.logsumexp(
            log_weights + log_survival_components,
            dim=-1
        )

        # uncensored = 0  -> dùng log S(t)
        # uncensored = 1  -> dùng log f(t)
        log_likelihood = torch.where(
            uncensored < 0.5,
            log_survival_mixture,
            log_pdf_mixture
        )

        negative_log_likelihood = -log_likelihood

        if reduction == "mean":
            return negative_log_likelihood.mean()

        if reduction == "sum":
            return negative_log_likelihood.sum()

        if reduction == "none":
            return negative_log_likelihood

        raise ValueError(
            f"reduction phải là 'mean', 'sum' hoặc 'none', "
            f"nhưng nhận được: {reduction}"
        )
            
def weibull_actual_parameters(
    output,
    eps=1e-7,
):
    """
    Chuyển hệ số Weibull của model thành tham số thực tế.

    output["x"]:
        [B, D]

    output["shapes"]:
        [B, M, D] — hệ số shape.

    output["scales"]:
        [B, M, D] — hệ số rate lambda.

    output["weights"]:
        [B, M]

    Returns
    -------
    shapes:
        [B, M], shape k > 0.

    rates:
        [B, M], rate lambda > 0.

    weights:
        [B, M], tổng theo M bằng 1.
    """

    x = output["x"].float()
    shape_coeffs = output["shapes"].float()
    rate_coeffs = output["scales"].float()
    weights = output["weights"].float()

    if x.ndim != 2:
        raise ValueError(
            f'output["x"] phải là [B, D], '
            f'nhận được {tuple(x.shape)}'
        )

    if shape_coeffs.ndim != 3:
        raise ValueError(
            f'output["shapes"] phải là [B, M, D], '
            f'nhận được {tuple(shape_coeffs.shape)}'
        )

    if rate_coeffs.shape != shape_coeffs.shape:
        raise ValueError(
            'output["scales"] phải cùng shape với '
            'output["shapes"]'
        )

    B, M, D = shape_coeffs.shape

    if x.shape != (B, D):
        raise ValueError(
            f'output["x"] phải là {(B, D)}, '
            f'nhận được {tuple(x.shape)}'
        )

    if weights.shape != (B, M):
        raise ValueError(
            f'output["weights"] phải là {(B, M)}, '
            f'nhận được {tuple(weights.shape)}'
        )

    # [B, D] và [B, M, D] -> [B, M]
    shape_logits = torch.einsum(
        "bd,bmd->bm",
        x,
        shape_coeffs,
    )

    rate_logits = torch.einsum(
        "bd,bmd->bm",
        x,
        rate_coeffs,
    )

    shapes = f.softplus(shape_logits) + eps
    rates = f.softplus(rate_logits) + eps

    if not torch.isfinite(weights).all():
        raise ValueError("weights chứa NaN hoặc Inf")

    if (weights < 0).any():
        raise ValueError("weights phải không âm")

    weight_sum = weights.sum(
        dim=-1,
        keepdim=True,
    )

    if (weight_sum <= 0).any():
        raise ValueError(
            "Mỗi bệnh nhân phải có tổng weights > 0"
        )

    weights = weights

    return shapes, rates, weights
def nll_survival_loss(
    logits,
    y,
    c,
    eps=1e-7,
    reduction="mean",
):
    """
    logits: [B, Nt]
    y:      [B] hoặc [B, 1]
    c:      [B] hoặc [B, 1]

    c = 0: quan sát sự kiện
    c = 1: censored
    """
    y = y.long().view(-1, 1)
    c = c.float().view(-1, 1)

    hazards = torch.sigmoid(logits)

    survival = torch.cumprod(
        1.0 - hazards,
        dim=1,
    )

    survival_padded = torch.cat(
        [
            torch.ones_like(
                survival[:, :1]
            ),
            survival,
        ],
        dim=1,
    )

    s_prev = survival_padded.gather(
        dim=1,
        index=y,
    ).clamp_min(eps)

    h_this = hazards.gather(
        dim=1,
        index=y,
    ).clamp(min=eps, max=1.0 - eps)

    s_this = survival_padded.gather(
        dim=1,
        index=y + 1,
    ).clamp_min(eps)

    event_loss = -(1.0 - c) * (
        torch.log(s_prev)
        + torch.log(h_this)
    )

    censored_loss = -c * torch.log(s_this)

    loss = event_loss + censored_loss

    if reduction == "mean":
        return loss.mean()

    if reduction == "sum":
        return loss.sum()

    if reduction == "none":
        return loss.squeeze(1)

    raise ValueError(
        f"reduction không hợp lệ: {reduction}"
    )

def get_KL_loss(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Tính KL divergence:

        KL(
            N(mu, diag(sigma^2))
            ||
            N(0, I)
        )

    Parameters
    ----------
    mu:
        Mean của các phân phối.
        Shape [..., latent_dim].

        Với toàn bộ proxy:
            [num_proxies, hidden_dim]

        Với positive proxy của batch:
            [batch_size, hidden_dim]

    sigma:
        Độ lệch chuẩn, không phải variance.
        Shape giống mu.

    eps:
        Giá trị nhỏ để tránh log(0).

    reduction:
        - "none": trả KL của từng phân phối, shape [...]
        - "mean": trung bình KL trên các phân phối
        - "sum": cộng KL trên các phân phối

    Returns
    -------
    Tensor KL scalar nếu reduction là mean/sum.
    """
    if mu.shape != sigma.shape:
        raise ValueError(
            f"mu và sigma phải cùng shape, "
            f"nhận được mu={tuple(mu.shape)}, "
            f"sigma={tuple(sigma.shape)}"
        )

    if not mu.is_floating_point():
        raise TypeError("mu phải là floating-point tensor.")

    if not sigma.is_floating_point():
        raise TypeError("sigma phải là floating-point tensor.")

    # sigma là standard deviation
    sigma = sigma.clamp_min(eps)

    variance = sigma.square()
    log_variance = 2.0 * torch.log(sigma)

    # KL theo từng chiều latent: [..., D]
    kl_per_dim = 0.5 * (
        mu.square()
        + variance
        - log_variance
        - 1.0
    )

    # Cộng trên latent dimension: [...]
    kl_per_distribution = kl_per_dim.sum(dim=-1)

    if reduction == "none":
        return kl_per_distribution

    if reduction == "mean":
        return kl_per_distribution.mean()

    if reduction == "sum":
        return kl_per_distribution.sum()

    raise ValueError(
        "reduction phải là 'none', 'mean' hoặc 'sum', "
        f"nhưng nhận được {reduction!r}."
    )


import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss cho survival analysis.

    Quy ước
    -------
    censor = 0:
        Event đã xảy ra.

    censor = 1:
        Right-censored.

    risk lớn hơn:
        Dự báo event có xu hướng xảy ra sớm hơn.

    Nếu risk = -RMST:
        risk_i > risk_j
        tương đương RMST_i < RMST_j.

    Cặp (i, j) so sánh được khi:
        1. i có event và time_i < time_j; hoặc
        2. i có event, j bị censor và time_i == time_j,
           nếu include_event_censor_ties=True.

    Loss xét tất cả cặp so sánh được.

    Trọng số:
        time_weight_strength > 0:
            Cặp có khoảng cách thời gian lớn hơn có trọng số lớn hơn.

    Margin:
        margin_ratio = 0:
            Không yêu cầu margin thay đổi theo thời gian.

        margin_ratio > 0:
            Khuyến khích chênh lệch risk lớn hơn khi
            khoảng cách thời gian quan sát lớn hơn.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        include_event_censor_ties: bool = True,
        time_scale: float = 1.0,
        time_weight_strength: float = 1.0,
        max_normalized_time_gap: float = 10.0,
        margin_ratio: float = 0.1,
        max_margin_time_gap: float | None = None,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError(
                "temperature phải > 0."
            )

        if time_scale <= 0:
            raise ValueError(
                "time_scale phải > 0."
            )

        if time_weight_strength < 0:
            raise ValueError(
                "time_weight_strength phải >= 0."
            )

        if max_normalized_time_gap <= 0:
            raise ValueError(
                "max_normalized_time_gap phải > 0."
            )

        if margin_ratio < 0:
            raise ValueError(
                "margin_ratio phải >= 0."
            )

        if (
            max_margin_time_gap is not None
            and max_margin_time_gap <= 0
        ):
            raise ValueError(
                "max_margin_time_gap phải > 0 hoặc None."
            )

        self.temperature = float(temperature)

        self.include_event_censor_ties = bool(
            include_event_censor_ties
        )

        self.time_scale = float(time_scale)

        self.time_weight_strength = float(
            time_weight_strength
        )

        self.max_normalized_time_gap = float(
            max_normalized_time_gap
        )

        self.margin_ratio = float(margin_ratio)

        self.max_margin_time_gap = (
            None
            if max_margin_time_gap is None
            else float(max_margin_time_gap)
        )

    def forward(
        self,
        risk: torch.Tensor,
        times: torch.Tensor,
        censor: torch.Tensor,
    ):
        """
        Parameters
        ----------
        risk:
            [B] hoặc [B, 1].
            Phải còn nằm trên computation graph.

        times:
            [B] hoặc [B, 1].
            Thời gian event/censor.

        censor:
            [B] hoặc [B, 1].
            0 = event, 1 = censored.

        Returns
        -------
        loss:
            Scalar tensor.

        num_pairs:
            Số cặp so sánh được.

        Lưu ý
        -----
        Interface trả về giống hàm cũ:
            loss, num_pairs = criterion(...)
        """

        risk = risk.reshape(-1)

        if not torch.is_floating_point(risk):
            raise TypeError(
                "risk phải là floating-point tensor."
            )

        # Khi dùng AMP hoặc float16, chuyển các phép tính loss
        # về ít nhất float32 để hạn chế vấn đề số học.
        #
        # .float() vẫn giữ gradient về risk ban đầu.
        if risk.dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            risk = risk.float()

        times = times.reshape(-1).to(
            device=risk.device,
        )

        censor = censor.reshape(-1).to(
            device=risk.device,
        )

        B = risk.numel()

        if times.numel() != B or censor.numel() != B:
            raise ValueError(
                "risk, times và censor phải có cùng số phần tử."
            )

        if not torch.isfinite(risk).all():
            raise ValueError(
                "risk chứa NaN hoặc Inf."
            )

        if not torch.isfinite(times).all():
            raise ValueError(
                "times chứa NaN hoặc Inf."
            )

        if (times < 0).any():
            raise ValueError(
                "times không được âm."
            )

        if not (
            (censor == 0)
            | (censor == 1)
        ).all():
            raise ValueError(
                "censor phải bằng 0 hoặc 1."
            )

        # Dùng dtype của risk cho phép tính khoảng cách.
        # Chuyển dtype vẫn giữ được gradient của risk.
        #
        # Nếu time có độ chính xác rất cao, quy tắc so sánh
        # thời gian nên được xác nhận theo dữ liệu gốc.
        time_i = times[:, None]
        time_j = times[None, :]

        censor_i = censor[:, None]
        censor_j = censor[None, :]

        # ---------------------------------------------------------
        # 1. Xác định cặp so sánh được.
        # ---------------------------------------------------------
        comparable = (
            (censor_i == 0)
            & (time_i < time_j)
        )

        # Đúng theo quy tắc tie trong hàm c_index em từng gửi:
        # i có event và j bị censor tại cùng thời gian.
        if self.include_event_censor_ties:
            comparable = comparable | (
                (censor_i == 0)
                & (censor_j == 1)
                & (time_i == time_j)
            )

        num_pairs = int(
            comparable.sum().item()
        )

        if num_pairs == 0:
            # Loss bằng 0 nhưng vẫn gắn với graph.
            return risk.sum() * 0.0, 0

        # ---------------------------------------------------------
        # 2. Chỉ lấy những cặp hợp lệ.
        # ---------------------------------------------------------
        risk_difference = (
            risk[:, None] - risk[None, :]
        )[comparable]

        # Với cặp hợp lệ, time_j >= time_i.
        time_gap = (
            time_j - time_i
        )[comparable].to(dtype=risk.dtype)

        # ---------------------------------------------------------
        # 3. Trọng số tăng theo khoảng cách thời gian.
        # ---------------------------------------------------------
        normalized_gap = (
            time_gap / self.time_scale
        ).clamp(
            max=self.max_normalized_time_gap,
        )

        pair_weights = (
            1.0
            + self.time_weight_strength * normalized_gap
        )

        # ---------------------------------------------------------
        # 4. Margin phụ thuộc khoảng cách thời gian.
        # ---------------------------------------------------------
        margin_gap = time_gap

        if self.max_margin_time_gap is not None:
            margin_gap = margin_gap.clamp(
                max=self.max_margin_time_gap,
            )

        margin = self.margin_ratio * margin_gap

        # Muốn:
        #
        # risk_i - risk_j >= margin
        #
        # Nếu risk = -RMST:
        #
        # RMST_j - RMST_i >= margin
        #
        # Đây là mục tiêu mềm: không phải ràng buộc cứng.
        pair_losses = F.softplus(
            (margin - risk_difference)
            / self.temperature
        )

        # ---------------------------------------------------------
        # 5. Weighted mean trên các cặp hợp lệ.
        # ---------------------------------------------------------
        loss = (
            pair_weights * pair_losses
        ).sum() / pair_weights.sum()

        return loss, num_pairs
def weibull_mixture_rmst(
    shapes: torch.Tensor,
    rates: torch.Tensor,
    weights: torch.Tensor,
    horizon: float,
    num_steps: int = 128,
):
    """
    Tính RMST của Weibull mixture bằng tích phân số.

    Parameters
    ----------
    shapes:
        [B, M], tham số shape k > 0.

    rates:
        [B, M], tham số rate lambda > 0 theo:
            S(t) = exp(-lambda * t**k)

    weights:
        [B, M], trọng số mixture không âm.

    horizon:
        Giới hạn tích phân, cùng đơn vị với thời gian.

    num_steps:
        Số điểm trên lưới tích phân, >= 2.

    Returns
    -------
    rmst:
        [B], có gradient đối với tham số đầu vào.
    """

    if horizon <= 0:
        raise ValueError(
            "horizon phải > 0."
        )

    if num_steps < 2:
        raise ValueError(
            "num_steps phải >= 2."
        )

    if (
        shapes.ndim != 2
        or rates.shape != shapes.shape
        or weights.shape != shapes.shape
    ):
        raise ValueError(
            "shapes, rates, weights phải cùng shape [B, M]."
        )

    if not (
        torch.isfinite(shapes).all()
        and torch.isfinite(rates).all()
        and torch.isfinite(weights).all()
    ):
        raise ValueError(
            "Tham số Weibull chứa NaN hoặc Inf."
        )

    if (shapes <= 0).any() or (rates <= 0).any():
        raise ValueError(
            "shapes và rates phải > 0."
        )

    if (weights < 0).any():
        raise ValueError(
            "weights phải không âm."
        )

    weight_sum = weights.sum(
        dim=-1,
        keepdim=True,
    )

    if (weight_sum <= 0).any():
        raise ValueError(
            "Mỗi bệnh nhân phải có tổng weights > 0."
        )

    weights = weights / weight_sum

    # Dùng float64 cho phép tích phân và log/exponential
    # để giảm nguy cơ tràn số so với float32.
    #
    # Việc chuyển dtype vẫn giữ gradient.
    shapes64 = shapes.to(torch.float64)
    rates64 = rates.to(torch.float64)
    weights64 = weights.to(torch.float64)

    # Điểm 0 được xử lý riêng:
    # S(0) = 1 cho Weibull có shape > 0.
    positive_times = torch.linspace(
        0.0,
        float(horizon),
        num_steps,
        device=shapes.device,
        dtype=torch.float64,
    )[1:]

    # [B, M, T-1]
    log_cumulative_hazard = (
        torch.log(rates64).unsqueeze(-1)
        + shapes64.unsqueeze(-1)
        * torch.log(positive_times)[None, None, :]
    )

    # Chặn giá trị lớn để tránh exp overflow.
    # Đây là xấp xỉ số: tại hazard rất lớn, S(t) gần 0.
    cumulative_hazard = torch.exp(
        log_cumulative_hazard.clamp(max=80.0)
    )

    component_survival = torch.exp(
        -cumulative_hazard
    )

    # [B, T-1]
    mixture_survival_positive = (
        weights64.unsqueeze(-1)
        * component_survival
    ).sum(dim=1)

    # Ghép S(0) = 1.
    survival_at_zero = torch.ones(
        shapes.size(0),
        1,
        device=shapes.device,
        dtype=torch.float64,
    )

    survival = torch.cat(
        [
            survival_at_zero,
            mixture_survival_positive,
        ],
        dim=1,
    )  # [B, T]

    times = torch.linspace(
        0.0,
        float(horizon),
        num_steps,
        device=shapes.device,
        dtype=torch.float64,
    )

    # Tích phân hình thang theo trục thời gian.
    rmst = torch.trapezoid(
        survival,
        x=times,
        dim=1,
    )

    return rmst  # [B], dtype float64

class PIBDWeibullLoss(nn.Module):
    """PIBD-inspired objective with Weibull NLL and optional expectation MSE."""

    def __init__(
        self,
        hidden_dim=128,
        survival_weight=2.0,
        proxy_weight=1.0,
        kl_weight=1e-2,
        alignment_weight=1.0,
        mi_weight=1.0,
        expectation_mse_weight=0.01,
        expectation_eps=1e-7,
        lambda_floor=0.01,
        lambda_penalty_weight=0.0,
        positive_triplet_weight=1.0,
        negative_triplet_weight=0.1,
        triplet_margin=0.1,
        ranking_weight=0.1,
    ):
        super().__init__()

        if expectation_mse_weight < 0:
            raise ValueError("expectation_mse_weight must be >= 0")

        self.weibull = WeibullLossandInference()

        self.survival_weight = survival_weight
        self.proxy_weight = proxy_weight
        self.kl_weight = kl_weight
        self.alignment_weight = alignment_weight
        self.mi_weight = mi_weight
        self.positive_triplet_weight = positive_triplet_weight
        self.negative_triplet_weight = negative_triplet_weight
        self.triplet_margin = triplet_margin
        self.ranking_weight = ranking_weight
        self.expectation_mse_weight = expectation_mse_weight
        self.expectation_eps = expectation_eps

        if lambda_floor <= 0:
            raise ValueError("lambda_floor must be > 0")

        if lambda_penalty_weight < 0:
            raise ValueError("lambda_penalty_weight must be >= 0")

        self.lambda_floor = float(lambda_floor)
        self.lambda_penalty_weight = float(lambda_penalty_weight)

        self.club_hp = CLUBStable(hidden_dim, hidden_dim)
        self.club_common = CLUBStable(2 * hidden_dim, hidden_dim)

        self.ranking_loss = PairwiseRankingLoss(temperature=1.0)


    def club_parameters(self):
        return list(self.club_hp.parameters()) + list(self.club_common.parameters())

    def club_learning_loss(self, output):
        h, p, g = (v.detach().float() for v in output['features'])
        return (self.club_hp.loss_loglikelihood(h, p)
                + self.club_common.loss_loglikelihood(torch.cat([h, p], -1), g))
    def expectation_mse(self, output, event_time, censor):
        """
        Tính MSE giữa kỳ vọng Weibull mixture và thời gian sự kiện.

        censor:
            0: có sự kiện
            1: censored

        Chỉ bệnh nhân có sự kiện được dùng trong MSE.
        """

        x = output["x"].float()
        event_time = event_time.reshape(-1).to(
            device=x.device,
            dtype=x.dtype,
        )
        censor = censor.reshape(-1).to(device=x.device)

        event_mask = censor == 0

        # Nếu batch toàn censored, MSE không đóng góp gradient.
        if not event_mask.any():
            return x.sum() * 0.0

        # Chỉ lấy bệnh nhân có sự kiện.
        x_event = x[event_mask]
        times_event = event_time[event_mask]

        shape_coeffs = output["shapes"][event_mask].float()
        rate_coeffs = output["scales"][event_mask].float()
        weights = output["weights"][event_mask].float()

        # [B_event, M, D] × [B_event, D] -> [B_event, M]
        shape_logits = torch.einsum(
            "bd,bmd->bm",
            x_event,
            shape_coeffs,
        )

        rate_logits = torch.einsum(
            "bd,bmd->bm",
            x_event,
            rate_coeffs,
        )

        shape = torch.nn.functional.softplus(
            shape_logits
        ) + self.expectation_eps

        rate = torch.nn.functional.softplus(
            rate_logits
        ) + self.expectation_eps

        weights = weights.clamp_min(self.expectation_eps)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # Quy ước: S_m(t) = exp(-rate_m * t**shape_m)
        #
        # E[T_m] = rate_m ** (-1 / shape_m)
        #          * Gamma(1 + 1 / shape_m)
        log_component_expectation = (
            torch.lgamma(1.0 + 1.0 / shape)
            - torch.log(rate) / shape
        )

        component_expectation = torch.exp(log_component_expectation)

        # [B_event, M] -> [B_event]
        predicted_expectation = (
            weights * component_expectation
        ).sum(dim=-1)

        if not torch.isfinite(predicted_expectation).all():
            raise FloatingPointError(
                "Weibull expectation contains NaN or Inf. "
                "Check shape and rate."
            )

        mse = torch.nn.functional.mse_loss(
            predicted_expectation,
            times_event,
            reduction="mean",
        )

        if not torch.isfinite(mse):
            raise FloatingPointError(
                "Expectation MSE contains NaN or Inf."
            )

        return mse
    def lambda_regularization(self, output):
        """
        Phạt khi Weibull rate lambda nhỏ hơn lambda_floor.

        output["x"]:      [B, D]
        output["scales"]: [B, M, D] — các hệ số, chưa phải lambda.
        """

        x = output["x"].float()
        rate_coeffs = output["scales"].float()

        rate_logits = torch.einsum(
            "bd,bmd->bm",
            x,
            rate_coeffs,
        )

        # Dùng cùng eps với WeibullLossandInference.
        rate = torch.nn.functional.softplus(rate_logits) + self.weibull.eps

        penalty = torch.relu(
            self.lambda_floor - rate
        ).square().mean()

        return penalty

    def forward(self, output, bins, event_time, censor):
        x = output['x'].float()
        bins = bins.reshape(-1).to(device=x.device, dtype=torch.long)
        censor = censor.reshape(-1).to(device=x.device, dtype=x.dtype)
        event_time = event_time.reshape(-1).to(device=x.device, dtype=x.dtype)
        if not ((censor == 0) | (censor == 1)).all():
            raise ValueError('censor must be 0 (event) or 1 (right-censored)')
        if not torch.isfinite(event_time).all() or (event_time < 0).any():
            raise ValueError('event_time must be finite and nonnegative')
        survival = self.weibull(x, event_time, 1-censor,
                               output['shapes'].float(), output['scales'].float(),
                               output['weights'].float())
        proxy = x.new_zeros(())
        kl = x.new_zeros(())
        alignment = x.new_zeros(())
        for name in ('histology', 'pathway'):
            aux = output[name]
            n_bins = aux['logits'].size(-1)
            if aux['logits'].size(0) != 2*n_bins:
                raise ValueError('Expected one proxy per (bin,censor) pair')
            proxy_bins = torch.arange(n_bins, device=x.device).repeat_interleave(2)
            proxy_censor = torch.tensor([0., 1.], device=x.device).repeat(n_bins)
            proxy = proxy + nll_survival_loss(aux['logits'].float(), proxy_bins, proxy_censor)
            kl = kl + get_KL_loss(aux['all_mu'].float(), aux['all_sigma'].float())
            # Minimize negative - positive similarity.
            alignment = alignment + (aux['negative'] - aux['positive']).mean()
        h, p, g = (v.float() for v in output['features'])
        # Freeze estimator parameters but retain gradients to encoder features.
        params = self.club_parameters()
        states = [v.requires_grad for v in params]
        try:
            for v in params:
                v.requires_grad_(False)
            mi = (self.club_hp(h, p)
                  + self.club_common(torch.cat([h, p], -1), g))
        finally:
            for v, state in zip(params, states):
                v.requires_grad_(state)

        positive_triplet = x.new_zeros(())
        negative_selected_triplet = x.new_zeros(())

        if (
            self.positive_triplet_weight > 0
            or self.negative_triplet_weight > 0
        ):
            for name in ("histology", "pathway"):
                triplet_data = output[name]["triplet"]

                # Tên key "anchor" là tên cũ trong dictionary.
                # Thực tế đây là vector proxy đúng.
                correct_proxy = triplet_data["anchor"]

                if self.positive_triplet_weight > 0:
                    loss_correct, n_correct = triplet_topk_correct_proxy(
                        topk_features=triplet_data["positives"],
                        feature_mask=triplet_data["positive_mask"],
                        correct_proxy=correct_proxy,
                        all_proxies=triplet_data["proxies"],
                        correct_proxy_idx=triplet_data["positive_proxy_idx"],
                        margin=self.triplet_margin,
                    )

                    positive_triplet = positive_triplet + loss_correct

                if self.negative_triplet_weight > 0:
                    loss_wrong, n_wrong = triplet_topk_wrong_proxy(
                        selected_features=(
                            triplet_data["negative_selected_features"]
                        ),
                        selected_mask=(
                            triplet_data["negative_selected_mask"]
                        ),
                        correct_proxy=correct_proxy,
                        wrong_proxies=(
                            triplet_data["negative_proxy_vectors"]
                        ),
                        margin=self.triplet_margin,
                    )

                    negative_selected_triplet = (
                        negative_selected_triplet + loss_wrong
                    )
        shapes, rates, weights = weibull_actual_parameters(
            output,
            eps=1e-7,
        )

        rmst = weibull_mixture_rmst(
            shapes=shapes,
            rates=rates,
            weights=weights,
            horizon=20,
            num_steps=128,
        )

        risk = -rmst

        ranking_loss, num_pairs = self.ranking_loss(
            risk=risk,
            times=event_time,
            censor=censor,
        )
        total_before_mse = (
            self.survival_weight * survival
            + self.proxy_weight * proxy
            + self.kl_weight * kl
            + self.alignment_weight * alignment
            + self.mi_weight * mi + self.lambda_penalty_weight * self.lambda_regularization(output) + self.positive_triplet_weight * positive_triplet + self.negative_triplet_weight * negative_selected_triplet
            + self.ranking_weight * ranking_loss
        )
        if self.expectation_mse_weight > 0:
            mse = self.expectation_mse(
                output=output,
                event_time=event_time,
                censor=censor,
            )
        else:
            # Không tính expectation khi tắt MSE.
            # Nhờ đó tránh tràn số không cần thiết.
            mse = x.new_zeros(())

        weighted_mse = self.expectation_mse_weight * mse
        total = total_before_mse + weighted_mse

        return {
            "total": total,
            "total_before_mse": total_before_mse,
            "survival": survival,
            "proxy": proxy,
            "kl": kl,
            "alignment": alignment,
            "mi": mi,
            "ranking_loss": ranking_loss,
            "expectation_mse": mse,
            "expectation_mse_weighted": weighted_mse,
            "positive_triplet": positive_triplet,
            "negative_selected_triplet": negative_selected_triplet,
            "positive_triplet_weighted": (
                self.positive_triplet_weight * positive_triplet
            ),
            "negative_selected_triplet_weighted": (
                self.negative_triplet_weight
                * negative_selected_triplet
            ),
        }




import torch
import torch.nn.functional as F


def triplet_topk_correct_proxy(
    topk_features,
    feature_mask,
    correct_proxy,
    all_proxies,
    correct_proxy_idx,
    margin=0.2,
):
    """
    Triplet từ top-k feature được chọn theo proxy đúng.

    Anchor:
        topk_features[b, k]

    Positive:
        correct_proxy[b]

    Negative:
        tất cả proxy p != correct_proxy_idx[b].

    Parameters
    ----------
    topk_features:
        [B, K, H]

    feature_mask:
        [B, K], True = feature thật, False = vị trí lặp/padding.

    correct_proxy:
        [B, H]

    all_proxies:
        [P, H]

    correct_proxy_idx:
        [B]

    Returns
    -------
    loss:
        Scalar tensor.

    n_triplets:
        Số triplet hợp lệ, kiểu Python int.
    """

    if margin < 0:
        raise ValueError("margin phải >= 0.")

    if topk_features.ndim != 3:
        raise ValueError("topk_features phải có shape [B, K, H].")

    B, K, H = topk_features.shape

    if correct_proxy.shape != (B, H):
        raise ValueError(
            f"correct_proxy cần shape {(B, H)}, "
            f"nhận được {tuple(correct_proxy.shape)}."
        )

    if all_proxies.ndim != 2 or all_proxies.size(1) != H:
        raise ValueError("all_proxies phải có shape [P, H].")

    P = all_proxies.size(0)

    if P < 2:
        raise ValueError("Cần ít nhất 2 proxy.")

    feature_mask = feature_mask.to(
        device=topk_features.device,
        dtype=torch.bool,
    )

    if feature_mask.shape != (B, K):
        raise ValueError(
            f"feature_mask cần shape {(B, K)}."
        )

    correct_proxy_idx = correct_proxy_idx.reshape(-1).to(
        device=topk_features.device,
        dtype=torch.long,
    )

    if correct_proxy_idx.shape != (B,):
        raise ValueError(
            f"correct_proxy_idx cần shape {(B,)}."
        )

    if (
        (correct_proxy_idx < 0).any()
        or (correct_proxy_idx >= P).any()
    ):
        raise ValueError("correct_proxy_idx không hợp lệ.")

    # Normalize để dùng cosine similarity.
    features = F.normalize(topk_features.float(), dim=-1)
    positive = F.normalize(correct_proxy.float(), dim=-1)
    proxies = F.normalize(all_proxies.float(), dim=-1)

    # sim(feature, proxy đúng): [B, K]
    sim_positive = (
        features * positive.unsqueeze(1)
    ).sum(dim=-1)

    # sim(feature, mọi proxy): [B, K, P]
    sim_all = torch.matmul(
        features,
        proxies.transpose(0, 1),
    )

    proxy_ids = torch.arange(
        P,
        device=features.device,
    ).view(1, 1, P)

    # Loại proxy đúng khỏi tập negative.
    negative_mask = (
        proxy_ids != correct_proxy_idx.view(B, 1, 1)
    )  # [B, 1, P]

    valid_mask = (
        feature_mask.unsqueeze(-1)
        & negative_mask
    )  # [B, K, P]

    # Cosine distance = 1 - similarity.
    #
    # d(feature, positive) - d(feature, negative)
    # = sim(feature, negative) - sim(feature, positive)
    values = F.relu(
        sim_all
        - sim_positive.unsqueeze(-1)
        + margin
    )

    n_triplets = int(valid_mask.sum().item())

    if n_triplets == 0:
        zero = (
            features.sum() + positive.sum() + proxies.sum()
        ) * 0.0
        return zero, 0

    return values[valid_mask].mean(), n_triplets


def triplet_topk_wrong_proxy(
    selected_features,
    selected_mask,
    correct_proxy,
    wrong_proxies,
    margin=0.2,
):
    """
    Triplet từ top-k feature được chọn theo từng proxy sai.

    Anchor:
        selected_features[b, q, k]

    Positive:
        correct_proxy[b]

    Negative:
        wrong_proxies[b, q] — proxy sai đã chọn feature đó.

    Parameters
    ----------
    selected_features:
        [B, Q, K, H], Q = P - 1.

    selected_mask:
        [B, Q, K], True = feature thật.

    correct_proxy:
        [B, H]

    wrong_proxies:
        [B, Q, H]

    Returns
    -------
    loss:
        Scalar tensor.

    n_triplets:
        Số triplet hợp lệ, kiểu Python int.
    """

    if margin < 0:
        raise ValueError("margin phải >= 0.")

    if selected_features.ndim != 4:
        raise ValueError(
            "selected_features phải có shape [B, Q, K, H]."
        )

    B, Q, K, H = selected_features.shape

    if correct_proxy.shape != (B, H):
        raise ValueError(
            f"correct_proxy cần shape {(B, H)}."
        )

    if wrong_proxies.shape != (B, Q, H):
        raise ValueError(
            f"wrong_proxies cần shape {(B, Q, H)}."
        )

    selected_mask = selected_mask.to(
        device=selected_features.device,
        dtype=torch.bool,
    )

    if selected_mask.shape != (B, Q, K):
        raise ValueError(
            f"selected_mask cần shape {(B, Q, K)}."
        )

    features = F.normalize(selected_features.float(), dim=-1)
    positive = F.normalize(correct_proxy.float(), dim=-1)
    negative = F.normalize(wrong_proxies.float(), dim=-1)

    # [B, Q, K]
    sim_positive = (
        features * positive[:, None, None, :]
    ).sum(dim=-1)

    # [B, Q, K]
    sim_negative = (
        features * negative[:, :, None, :]
    ).sum(dim=-1)

    values = F.relu(
        sim_negative - sim_positive + margin
    )

    n_triplets = int(selected_mask.sum().item())

    if n_triplets == 0:
        zero = (
            features.sum() + positive.sum() + negative.sum()
        ) * 0.0
        return zero, 0

    return values[selected_mask].mean(), n_triplets