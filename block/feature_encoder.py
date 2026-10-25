from collections.abc import Sequence

import torch
import torch.nn as nn
from .blocks import ResidualBlock, Proxies

class FeatureEncoder(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, num_blocks:int, dropout:float=0.1):
        super(FeatureEncoder, self).__init__()
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
    def forward(self, x):
        out = self.input_linear(x)
        for block in self.blocks:
            out = block(out)
        return out
    
class ModalEncoder(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, num_blocks:int, dropout:float, num_proxies:int, top_k:int, sample_num:int, seed:int, beta:float):
        super(ModalEncoder, self).__init__()
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.proxies = Proxies(dim=hidden_dim, num_proxies=num_proxies)
        self.decoder = nn.Linear(hidden_dim, num_proxies)
        self.num_proxies = num_proxies
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.sample_num = sample_num
        self.beta = beta
        self.seed = seed
        self.dropout = dropout
    def gaussian_noise_train(self):
        return torch.normal(torch.zeros(self.num_proxies, self.sample_num, self.hidden_dim), torch.ones(self.num_proxies, self.sample_num, self.hidden_dim))
    def gaussian_noise_inference(self):
        return torch.normal(torch.zeros(self.num_proxies, self.sample_num, self.hidden_dim), torch.ones(self.num_proxies, self.sample_num, self.hidden_dim), generator= torch.manual_seed(self.seed))
    def forward(self, x, y, c,  mask = None ):
        histfeatures = self.input_linear(x)
        for block in self.encoder:
            histfeatures = block(histfeatures)
        mu_proxies, sigma_proxies = self.proxies()
        noise = self.gaussian_noise_train().to(device= mu_proxies.device, dtype=mu_proxies.dtype)
        sampled_proxies = mu_proxies.unsqueeze(1) + sigma_proxies.unsqueeze(1) * noise
        mean_proxies = nn.functional.normalize(torch.mean(sampled_proxies, dim=1), dim=-1).unsqueeze(0)
        histfeatures = nn.functional.normalize(histfeatures, dim=-1)
        similarity = torch.matmul(histfeatures, mean_proxies.transpose(1, 2))
        B, N, P = similarity.shape
        if mask is None:
            padding_mask = torch.zeros((B, N), dtype=torch.bool, device=similarity.device)
        else:
            padding_mask = mask.to(
                device=similarity.device,
                dtype=torch.bool
            )
        valid_patch_mask = ~padding_mask
        if padding_mask.shape != (B, N):
            raise ValueError(
                f"mask phải có shape {(B, N)}, "
                f"nhưng nhận được {tuple(padding_mask.shape)}"
            )
        similarity = similarity.masked_fill(
        padding_mask.unsqueeze(-1),
        float("-inf")
        )   
        y = y.reshape(-1).to(device=x.device, dtype=torch.long)
        c = c.reshape(-1).to(device=x.device, dtype=torch.long)

        proxies_indice = self.proxies.proxy_lookup[y, c]
        batch_idx = torch.arange(x.size(0), device=x.device)
        similarity_positive = similarity[batch_idx, :, proxies_indice].unsqueeze(-1)
        all_proxy_indices = torch.arange(
            P,
            device=similarity.device
        ).unsqueeze(0).expand(B, P)
        negative_indices = all_proxy_indices[
            all_proxy_indices != proxies_indice.unsqueeze(1)
        ].view(B, P - 1)
        # Mở rộng theo N feature
        negative_indices = negative_indices.unsqueeze(1).expand(
            B, N, P - 1
        )
        similarity_negative = torch.gather(
            similarity,
            dim=2,
            index=negative_indices
        )
        similarity_topk_positive, index_p = torch.topk(similarity_positive.squeeze(-1), self.top_k, dim = 1)
        similarity_topk_negative, index_n = torch.topk(similarity_negative, self.top_k, dim = 1)
        mask_positive = torch.gather(input=valid_patch_mask, dim=1, index=index_p)
        mask_negative = torch.gather(input=valid_patch_mask.unsqueeze(-1).expand(-1, -1, similarity_topk_negative.size(-1)), dim=1, index=index_n)
        positive = similarity_topk_positive.masked_fill(
            ~mask_positive, 0.0
        )
        negative = similarity_topk_negative.masked_fill(
            ~mask_negative, 0.0
        )

        similarity_positive_mean = (
            positive.sum(dim=1)
            / mask_positive.sum(dim=1).clamp_min(1)
        )
        similarity_negative_mean = (
            negative.sum(dim=(1, 2))
            / mask_negative.sum(dim=(1, 2)).clamp_min(1)
        )
        #similarity_positive_mean = torch.mean(similarity_topk_positive * mask_positive, dim=1)
        #similarity_negative_mean = torch.mean(similarity_topk_negative * mask_negative, dim=(1,2))
        mu_true = mu_proxies[proxies_indice]
        sigma_true = sigma_proxies[proxies_indice]
        similarity_unbind = similarity.view(similarity.size(0), -1)
        _, similarity_topk_idx = torch.topk(similarity_unbind, self.top_k, dim=1)
        similarity_topk_idx = torch.div(similarity_topk_idx, self.num_proxies, rounding_mode='floor')
        #similarity_topk_idx = similarity_topk_idx % similarity.size(1)
        gather_index = similarity_topk_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        histfeatures_topk = torch.gather(histfeatures, 1, gather_index)
        logits = torch.mean(self.decoder(sampled_proxies), dim = 1)
        return similarity_positive_mean, similarity_negative_mean, mu_true, sigma_true, mu_proxies, sigma_proxies, histfeatures_topk, logits
    def predict(self, x):
        histfeatures = self.input_linear(x)
        for block in self.encoder:
            histfeatures = block(histfeatures)
        mu_proxies, sigma_proxies = self.proxies()
        noise = self.gaussian_noise_inference().to(device=mu_proxies.device, dtype=mu_proxies.dtype)
        sampled_proxies = mu_proxies.unsqueeze(1) + sigma_proxies.unsqueeze(1) * noise
        mean_proxies = nn.functional.normalize(torch.mean(sampled_proxies, dim=1), dim=-1).unsqueeze(0)
        histfeatures = nn.functional.normalize(histfeatures, dim=-1)
        similarity = torch.matmul(histfeatures, mean_proxies.transpose(1, 2))
        similarity_unbind = similarity.view(similarity.size(0), -1)
        _, similarity_topk_idx = torch.topk(similarity_unbind, self.top_k, dim=1)
        proxies_topk_idx = similarity_topk_idx % self.num_proxies
        similarity_topk_idx = torch.div(similarity_topk_idx, self.num_proxies, rounding_mode='floor')
        proxies_indice = torch.mode(proxies_topk_idx, dim =1).values
        mu_true = mu_proxies[proxies_indice]
        sigma_true = sigma_proxies[proxies_indice]
        gather_index = similarity_topk_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        histfeatures_topk = torch.gather(histfeatures, 1, gather_index)
        logits = torch.mean(self.decoder(sampled_proxies), dim = 1)
        return mu_true, sigma_true,  mu_proxies, sigma_proxies, histfeatures_topk, logits
    
def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim,
    keepdim: bool = False,
):
    """
    Tính mean chỉ trên các phần tử hợp lệ.

    values:
        Tensor chứa dữ liệu, có thể có -inf hoặc nan.

    mask:
        True  = phần tử hợp lệ
        False = phần tử cần bỏ qua.

    dim:
        Chiều cần tính mean.
    """

    mask = mask.to(
        device=values.device,
        dtype=torch.bool
    )

    # Broadcast mask về cùng shape với values
    mask = torch.broadcast_to(mask, values.shape)

    # Đồng thời loại bỏ inf, -inf và nan
    finite_mask = torch.isfinite(values)
    final_mask = mask & finite_mask

    # Không dùng values * final_mask vì -inf * 0 = nan
    safe_values = torch.where(
        final_mask,
        values,
        torch.zeros_like(values)
    )

    value_sum = safe_values.sum(
        dim=dim,
        keepdim=keepdim
    )

    valid_count = final_mask.sum(
        dim=dim,
        keepdim=keepdim
    )

    mean = value_sum / valid_count.clamp_min(1)

    return mean, valid_count

class SharedMultiGroupEncoder(nn.Module):
    def __init__(
        self,
        group_sizes: Sequence[int],
        hidden_dim: int,
        num_blocks: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        if len(group_sizes) == 0:
            raise ValueError(
                "group_sizes không được rỗng"
            )

        self.group_sizes = tuple(group_sizes)
        self.num_groups = len(group_sizes)
        self.hidden_dim = hidden_dim

        # Mỗi nhóm cần input projection riêng
        self.input_projections = nn.ModuleList([
            nn.Linear(
                input_dim,
                hidden_dim,
            )
            for input_dim in self.group_sizes
        ])

        # Dùng chung residual backbone
        self.shared_blocks = nn.ModuleList([
            ResidualBlock(
                hidden_dim,
                dropout,
            )
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        x_groups: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(x_groups) != self.num_groups:
            raise ValueError(
                f"Expected {self.num_groups} groups, "
                f"got {len(x_groups)}"
            )

        projected_groups = []

        for index, (
            projection,
            group_features,
            expected_size,
        ) in enumerate(
            zip(
                self.input_projections,
                x_groups,
                self.group_sizes,
            )
        ):
            if group_features.ndim != 2:
                raise ValueError(
                    f"x_groups[{index}] phải là [B,D]"
                )

            if group_features.size(1) != expected_size:
                raise ValueError(
                    f"x_groups[{index}] cần "
                    f"{expected_size} features, "
                    f"nhận được "
                    f"{group_features.size(1)}"
                )

            # [B,input_dim_i] → [B,H]
            projected = projection(
                group_features
            )

            projected_groups.append(projected)

        # [B,M,H]
        out = torch.stack(
            projected_groups,
            dim=1,
        )

        # ResidualBlock sử dụng Linear trên chiều cuối,
        # vì vậy xử lý trực tiếp được tensor [B,M,H].
        for block in self.shared_blocks:
            out = block(out)

        return out
