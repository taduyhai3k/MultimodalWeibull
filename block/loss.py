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

        # Chuẩn hóa lại để tổng trọng số bằng 1
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
        total_before_mse = (
            self.survival_weight * survival
            + self.proxy_weight * proxy
            + self.kl_weight * kl
            + self.alignment_weight * alignment
            + self.mi_weight * mi + self.lambda_penalty_weight * self.lambda_regularization(output)
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
            "expectation_mse": mse,
            "expectation_mse_weighted": weighted_mse,
        }