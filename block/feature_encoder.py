from collections.abc import Sequence

import torch
import torch.nn as nn
from .blocks import ResidualBlock, Proxies
import torch.nn.functional as F

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
    
# class ModalEncoder(nn.Module):
#     def __init__(self, input_dim:int, hidden_dim:int, num_blocks:int, dropout:float, num_proxies:int, top_k:int, sample_num:int, seed:int, beta:float):
#         super(ModalEncoder, self).__init__()
#         self.input_linear = nn.Linear(input_dim, hidden_dim)
#         self.encoder = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
#         self.proxies = Proxies(dim=hidden_dim, num_proxies=num_proxies)
#         self.decoder = nn.Linear(hidden_dim, num_proxies)
#         self.num_proxies = num_proxies
#         self.hidden_dim = hidden_dim
#         self.top_k = top_k
#         self.sample_num = sample_num
#         self.beta = beta
#         self.seed = seed
#         self.dropout = dropout
#     def gaussian_noise_train(self):
#         return torch.normal(torch.zeros(self.num_proxies, self.sample_num, self.hidden_dim), torch.ones(self.num_proxies, self.sample_num, self.hidden_dim))
#     def gaussian_noise_inference(self):
#         return torch.normal(torch.zeros(self.num_proxies, self.sample_num, self.hidden_dim), torch.ones(self.num_proxies, self.sample_num, self.hidden_dim), generator= torch.manual_seed(self.seed))
#     def forward(self, x, y, c,  mask = None ):
#         histfeatures = self.input_linear(x)
#         for block in self.encoder:
#             histfeatures = block(histfeatures)
#         mu_proxies, sigma_proxies = self.proxies()
#         noise = self.gaussian_noise_train().to(device= mu_proxies.device, dtype=mu_proxies.dtype)
#         sampled_proxies = mu_proxies.unsqueeze(1) + sigma_proxies.unsqueeze(1) * noise
#         mean_proxies = nn.functional.normalize(torch.mean(sampled_proxies, dim=1), dim=-1).unsqueeze(0)
#         histfeatures = nn.functional.normalize(histfeatures, dim=-1)
#         similarity = torch.matmul(histfeatures, mean_proxies.transpose(1, 2))
#         B, N, P = similarity.shape
#         if mask is None:
#             padding_mask = torch.zeros((B, N), dtype=torch.bool, device=similarity.device)
#         else:
#             padding_mask = mask.to(
#                 device=similarity.device,
#                 dtype=torch.bool
#             )
#         valid_patch_mask = ~padding_mask
#         if padding_mask.shape != (B, N):
#             raise ValueError(
#                 f"mask phải có shape {(B, N)}, "
#                 f"nhưng nhận được {tuple(padding_mask.shape)}"
#             )
#         similarity = similarity.masked_fill(
#         padding_mask.unsqueeze(-1),
#         float("-inf")
#         )   
#         y = y.reshape(-1).to(device=x.device, dtype=torch.long)
#         c = c.reshape(-1).to(device=x.device, dtype=torch.long)

#         proxies_indice = self.proxies.proxy_lookup[y, c]
#         batch_idx = torch.arange(x.size(0), device=x.device)
#         similarity_positive = similarity[batch_idx, :, proxies_indice].unsqueeze(-1)
#         all_proxy_indices = torch.arange(
#             P,
#             device=similarity.device
#         ).unsqueeze(0).expand(B, P)
#         negative_indices = all_proxy_indices[
#             all_proxy_indices != proxies_indice.unsqueeze(1)
#         ].view(B, P - 1)
#         # Mở rộng theo N feature
#         negative_indices = negative_indices.unsqueeze(1).expand(
#             B, N, P - 1
#         )
#         similarity_negative = torch.gather(
#             similarity,
#             dim=2,
#             index=negative_indices
#         )
#         similarity_topk_positive, index_p = torch.topk(similarity_positive.squeeze(-1), self.top_k, dim = 1)
#         similarity_topk_negative, index_n = torch.topk(similarity_negative, self.top_k, dim = 1)
#         mask_positive = torch.gather(input=valid_patch_mask, dim=1, index=index_p)
#         mask_negative = torch.gather(input=valid_patch_mask.unsqueeze(-1).expand(-1, -1, similarity_topk_negative.size(-1)), dim=1, index=index_n)
#         positive = similarity_topk_positive.masked_fill(
#             ~mask_positive, 0.0
#         )
#         negative = similarity_topk_negative.masked_fill(
#             ~mask_negative, 0.0
#         )

#         similarity_positive_mean = (
#             positive.sum(dim=1)
#             / mask_positive.sum(dim=1).clamp_min(1)
#         )
#         similarity_negative_mean = (
#             negative.sum(dim=(1, 2))
#             / mask_negative.sum(dim=(1, 2)).clamp_min(1)
#         )
#         #similarity_positive_mean = torch.mean(similarity_topk_positive * mask_positive, dim=1)
#         #similarity_negative_mean = torch.mean(similarity_topk_negative * mask_negative, dim=(1,2))
#         mu_true = mu_proxies[proxies_indice]
#         sigma_true = sigma_proxies[proxies_indice]
#         similarity_unbind = similarity.view(similarity.size(0), -1)
#         _, similarity_topk_idx = torch.topk(similarity_unbind, self.top_k, dim=1)
#         similarity_topk_idx = torch.div(similarity_topk_idx, self.num_proxies, rounding_mode='floor')
#         #similarity_topk_idx = similarity_topk_idx % similarity.size(1)
#         gather_index = similarity_topk_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
#         histfeatures_topk = torch.gather(histfeatures, 1, gather_index)
#         logits = torch.mean(self.decoder(sampled_proxies), dim = 1)
#         return similarity_positive_mean, similarity_negative_mean, mu_true, sigma_true, mu_proxies, sigma_proxies, histfeatures_topk, logits
#     def predict(self, x):
#         histfeatures = self.input_linear(x)
#         for block in self.encoder:
#             histfeatures = block(histfeatures)
#         mu_proxies, sigma_proxies = self.proxies()
#         noise = self.gaussian_noise_inference().to(device=mu_proxies.device, dtype=mu_proxies.dtype)
#         sampled_proxies = mu_proxies.unsqueeze(1) + sigma_proxies.unsqueeze(1) * noise
#         mean_proxies = nn.functional.normalize(torch.mean(sampled_proxies, dim=1), dim=-1).unsqueeze(0)
#         histfeatures = nn.functional.normalize(histfeatures, dim=-1)
#         similarity = torch.matmul(histfeatures, mean_proxies.transpose(1, 2))
#         similarity_unbind = similarity.view(similarity.size(0), -1)
#         _, similarity_topk_idx = torch.topk(similarity_unbind, self.top_k, dim=1)
#         proxies_topk_idx = similarity_topk_idx % self.num_proxies
#         similarity_topk_idx = torch.div(similarity_topk_idx, self.num_proxies, rounding_mode='floor')
#         proxies_indice = torch.mode(proxies_topk_idx, dim =1).values
#         mu_true = mu_proxies[proxies_indice]
#         sigma_true = sigma_proxies[proxies_indice]
#         gather_index = similarity_topk_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
#         histfeatures_topk = torch.gather(histfeatures, 1, gather_index)
#         logits = torch.mean(self.decoder(sampled_proxies), dim = 1)
#         return mu_true, sigma_true,  mu_proxies, sigma_proxies, histfeatures_topk, logits
    



class ModalEncoder(nn.Module):
    """
    Encoder cho một modality: histology hoặc pathway.

    Quy ước mask:
        False = token thật
        True  = padding

    Forward:
        Chọn top-k token theo proxy đúng từ (y, c).

    Predict:
        Không sử dụng nhãn.
        Chọn proxy dự đoán từ similarity, sau đó chọn top-k
        token theo proxy dự đoán.

    Đầu ra forward mặc định:
        8 tensor giống thứ tự cũ.

    Nếu return_triplet=True:
        Trả thêm dictionary triplet ở vị trí thứ 9.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        dropout: float,
        num_proxies: int,
        top_k: int,
        sample_num: int,
        seed: int,
        beta: float,
    ):
        super().__init__()

        if num_proxies < 2 or num_proxies % 2 != 0:
            raise ValueError(
                "num_proxies phải là số chẵn và >= 2 "
                "với proxy_lookup hiện tại."
            )

        if top_k <= 0:
            raise ValueError("top_k phải > 0.")

        if sample_num <= 0:
            raise ValueError("sample_num phải > 0.")

        self.input_linear = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.encoder = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(num_blocks)
        ])

        self.proxies = Proxies(
            dim=hidden_dim,
            num_proxies=num_proxies,
        )

        # Giữ nguyên output dimension như code cũ của em.
        self.decoder = nn.Linear(
            hidden_dim,
            num_proxies,
        )

        self.num_proxies = num_proxies
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.sample_num = sample_num
        self.seed = seed
        self.beta = beta
        self.dropout = dropout

    def _encode(self, x):
        """
        x: [B, N, input_dim]
        return: [B, N, hidden_dim], đã L2 normalize.
        """
        if x.ndim != 3:
            raise ValueError(
                f"x phải có shape [B, N, D], "
                f"nhận được {tuple(x.shape)}"
            )

        if x.size(0) == 0 or x.size(1) == 0:
            raise ValueError("Batch và số token phải > 0.")

        features = self.input_linear(x)

        for block in self.encoder:
            features = block(features)

        return F.normalize(features, p=2, dim=-1)

    def _sample_proxies(self, deterministic=False):
        """
        Sinh mẫu proxy bằng reparameterization.

        deterministic=False:
            Dùng RNG hiện tại cho training.

        deterministic=True:
            Dùng generator riêng với seed cố định cho inference;
            không thay đổi seed toàn cục.
        """
        mu_proxies, sigma_proxies = self.proxies()

        generator = None

        if deterministic:
            generator = torch.Generator(device=mu_proxies.device)
            generator.manual_seed(self.seed)

        noise = torch.randn(
            self.num_proxies,
            self.sample_num,
            self.hidden_dim,
            device=mu_proxies.device,
            dtype=mu_proxies.dtype,
            generator=generator,
        )

        sampled_proxies = (
            mu_proxies.unsqueeze(1)
            + sigma_proxies.unsqueeze(1) * noise
        )  # [P, S, H]

        mean_proxies = F.normalize(
            sampled_proxies.mean(dim=1),
            p=2,
            dim=-1,
        )  # [P, H]

        logits = self.decoder(
            sampled_proxies
        ).mean(dim=1)  # [P, num_proxies]

        return (
            mu_proxies,
            sigma_proxies,
            sampled_proxies,
            mean_proxies,
            logits,
        )

    @staticmethod
    def _padding_mask(x, mask):
        """
        Trả bool mask [B, N]:
            True = padding.
        """
        B, N = x.shape[:2]

        if mask is None:
            padding_mask = torch.zeros(
                B,
                N,
                dtype=torch.bool,
                device=x.device,
            )
        else:
            padding_mask = mask.to(
                device=x.device,
                dtype=torch.bool,
            )

            if padding_mask.shape != (B, N):
                raise ValueError(
                    f"mask phải có shape {(B, N)}, "
                    f"nhận được {tuple(padding_mask.shape)}."
                )

        valid_counts = (~padding_mask).sum(dim=1)

        if (valid_counts == 0).any():
            bad_rows = torch.nonzero(
                valid_counts == 0,
                as_tuple=False,
            ).flatten().tolist()

            raise ValueError(
                "Mỗi bệnh nhân phải có ít nhất một token thật. "
                f"Batch indices không hợp lệ: {bad_rows}"
            )

        return padding_mask

    def _label_to_proxy(self, y, c, batch_size, device):
        """
        Quy ước proxy_lookup:
            y = bin thời gian
            c = 0 hoặc 1
        """
        y = y.reshape(-1).to(device=device, dtype=torch.long)
        c = c.reshape(-1).to(device=device, dtype=torch.long)

        if y.numel() != batch_size or c.numel() != batch_size:
            raise ValueError(
                "y và c phải có đúng một giá trị cho mỗi bệnh nhân."
            )

        lookup = self.proxies.proxy_lookup

        if ((y < 0) | (y >= lookup.size(0))).any():
            raise ValueError(
                f"y phải nằm trong [0, {lookup.size(0) - 1}]. "
                "Kiểm tra y có phải bin thời gian hay không."
            )

        if ((c < 0) | (c >= lookup.size(1))).any():
            raise ValueError(
                "c phải tương ứng với cột của proxy_lookup "
                "(thông thường 0 hoặc 1)."
            )

        return lookup[y, c]  # [B]

    def _topk_by_proxy(
        self,
        features,
        similarity,
        proxy_idx,
        padding_mask,
    ):
        """
        Chọn feature theo một proxy cho từng bệnh nhân.

        features:     [B, N, H]
        similarity:   [B, N, P]
        proxy_idx:    [B]
        padding_mask: [B, N], True = padding

        Returns
        -------
        selected:
            [B, top_k, H], nếu thiếu token thì lặp token thật.

        selected_mask:
            [B, top_k], True = token được chọn hợp lệ,
            False = vị trí được lặp để đủ kích thước.

        top_values:
            [B, min(top_k, N)], similarity; có thể chứa -inf
            tại vị trí padding.

        raw_valid:
            [B, min(top_k, N)], True nếu index tương ứng token thật.
        """
        B, N, H = features.shape
        K0 = min(self.top_k, N)

        batch_idx = torch.arange(
            B,
            device=features.device,
        )

        selected_similarity = similarity[
            batch_idx, :, proxy_idx
        ]  # [B, N]

        masked_similarity = selected_similarity.masked_fill(
            padding_mask,
            float("-inf"),
        )

        top_values, top_indices = torch.topk(
            masked_similarity,
            k=K0,
            dim=1,
        )

        raw_valid = torch.gather(
            ~padding_mask,
            dim=1,
            index=top_indices,
        )  # [B, K0]

        # Mỗi bệnh nhân đã được kiểm tra có ít nhất 1 token thật.
        # Thay index padding bằng token thật đầu tiên trong top-k.
        fallback = top_indices[:, :1].expand_as(top_indices)

        safe_indices = torch.where(
            raw_valid,
            top_indices,
            fallback,
        )

        # Nếu top_k > N, bổ sung thêm bản sao của token thật.
        if self.top_k > K0:
            extra_count = self.top_k - K0

            extra_indices = top_indices[:, :1].expand(
                B,
                extra_count,
            )

            safe_indices = torch.cat(
                [safe_indices, extra_indices],
                dim=1,
            )

            extra_valid = torch.zeros(
                B,
                extra_count,
                dtype=torch.bool,
                device=features.device,
            )

            selected_mask = torch.cat(
                [raw_valid, extra_valid],
                dim=1,
            )
        else:
            selected_mask = raw_valid

        selected = torch.gather(
            features,
            dim=1,
            index=safe_indices.unsqueeze(-1).expand(
                -1, -1, H
            ),
        )

        return selected, selected_mask, top_values, raw_valid

    def forward(
        self,
        x,
        y,
        c,
        mask=None,
        return_triplet=True,
    ):
        """
        Training: chọn top-k theo proxy đúng.

        y:
            Bin thời gian, không phải label_discrete = 2*y+c.

        c:
            Censor: 0 hoặc 1.

        Nếu return_triplet=True, trả thêm dictionary để tính loss.
        """
        features = self._encode(x)

        B, N, H = features.shape
        padding_mask = self._padding_mask(x, mask)

        (
            mu_proxies,
            sigma_proxies,
            _,
            mean_proxies,
            logits,
        ) = self._sample_proxies(deterministic=False)

        # [B, N, H] @ [H, P] -> [B, N, P]
        similarity = features @ mean_proxies.transpose(0, 1)

        proxy_idx = self._label_to_proxy(
            y,
            c,
            batch_size=B,
            device=x.device,
        )

        # ---------------------------------------------------------
        # 1. Chọn top-k feature theo PROXY ĐÚNG.
        # ---------------------------------------------------------
        (
            _features_by_true_proxy,
            _features_by_true_proxy_mask,
            positive_values,
            positive_valid,
        ) = self._topk_by_proxy(
            features=features,
            similarity=similarity,
            proxy_idx=proxy_idx,
            padding_mask=padding_mask,
        )

        safe_positive = positive_values.masked_fill(
            ~positive_valid,
            0.0,
        )

        positive_mean = (
            safe_positive.sum(dim=1)
            / positive_valid.sum(dim=1).clamp_min(1)
        )

        # ---------------------------------------------------------
        # 2. Giữ cách tính negative mean theo từng proxy sai.
        # ---------------------------------------------------------
        K0 = min(self.top_k, N)

        proxy_ids = torch.arange(
            self.num_proxies,
            device=x.device,
        ).unsqueeze(0).expand(B, -1)

        (
            features_topk,
            features_topk_mask,
            global_token_indices,
            global_proxy_indices,
            global_similarity_values,
        ) = self._topk_across_proxies(
            features=features,
            similarity=similarity,
            padding_mask=padding_mask,
        )

        negative_indices = proxy_ids[
            proxy_ids != proxy_idx.unsqueeze(1)
        ].reshape(B, self.num_proxies - 1)

        negative_similarity = torch.gather(
            similarity,
            dim=2,
            index=negative_indices.unsqueeze(1).expand(
                B,
                N,
                self.num_proxies - 1,
            ),
        )  # [B, N, P-1]

        negative_similarity = negative_similarity.masked_fill(
            padding_mask.unsqueeze(-1),
            float("-inf"),
        )

        negative_values, negative_token_idx = torch.topk(
            negative_similarity,
            k=K0,
            dim=1,
        )

        valid_tokens = (~padding_mask).unsqueeze(-1).expand(
            B,
            N,
            self.num_proxies - 1,
        )

        negative_valid = torch.gather(
            valid_tokens,
            dim=1,
            index=negative_token_idx,
        )

        safe_negative = negative_values.masked_fill(
            ~negative_valid,
            0.0,
        )

        negative_mean = (
            safe_negative.sum(dim=(1, 2))
            / negative_valid.sum(dim=(1, 2)).clamp_min(1)
        )  # [B]

        mu_true = mu_proxies[proxy_idx]
        sigma_true = sigma_proxies[proxy_idx]

        base_output = (
            positive_mean,
            negative_mean,
            mu_true,
            sigma_true,
            mu_proxies,
            sigma_proxies,
            features_topk,
            logits,
        )

        # ---------------------------------------------------------
        # Top-k feature tương ứng với từng proxy sai
        # ---------------------------------------------------------

        num_negative_proxies = self.num_proxies - 1

        # [B, K0, P-1] -> [B, (P-1)*K0]
        negative_token_idx_flat = (
            negative_token_idx
            .permute(0, 2, 1)
            .reshape(B, num_negative_proxies * K0)
        )

        # [B, N, H] -> lấy các feature theo index.
        negative_features_flat = torch.gather(
            features,
            dim=1,
            index=negative_token_idx_flat.unsqueeze(-1).expand(
                -1,
                -1,
                self.hidden_dim,
            ),
        )

        # [B, P-1, K0, H]
        negative_selected_features = negative_features_flat.reshape(
            B,
            num_negative_proxies,
            K0,
            self.hidden_dim,
        )

        # [B, K0, P-1] -> [B, P-1, K0]
        negative_selected_mask = negative_valid.permute(0, 2, 1)

        # Lấy vector của từng proxy sai theo đúng thứ tự negative_indices.
        # mean_proxies: [P, H]
        # negative_indices: [B, P-1]
        negative_proxy_vectors = mean_proxies[negative_indices]
        # [B, P-1, H]

        if not return_triplet:
            return base_output

        triplet_data = {
            # Proxy đúng: positive trong triplet loss.
            # Giữ tên "anchor" để tương thích với code loss hiện tại.
            "anchor": mean_proxies[proxy_idx],  # [B, H]

            # Anchor thực sự của triplet:
            # top-k feature được chọn trên toàn bộ proxy.
            "positives": features_topk,                 # [B, K, H]
            "positive_mask": features_topk_mask,         # [B, K]

            # Toàn bộ proxy và index proxy đúng.
            "proxies": mean_proxies,                     # [P, H]
            "positive_proxy_idx": proxy_idx,             # [B]

            # Giữ nguyên: top-k feature theo từng proxy sai.
            "negative_selected_features": (
                negative_selected_features
            ),  # [B, P-1, K0, H]

            "negative_selected_mask": (
                negative_selected_mask
            ),  # [B, P-1, K0]

            "negative_proxy_vectors": (
                negative_proxy_vectors
            ),  # [B, P-1, H]

            # Thông tin top-k toàn bộ proxy.
            "global_features_mask": features_topk_mask,
            "global_token_indices": global_token_indices,
            "global_proxy_indices": global_proxy_indices,
            "global_similarity_values": global_similarity_values,
        }

        return (*base_output, triplet_data)

    def predict(
        self,
        x,
        mask=None,
        return_details=False,
    ):
        """
        Inference không sử dụng y hoặc c.

        Cách chọn:
            1. Chọn top-k cặp (feature, proxy) trên toàn bộ proxy.
            2. Lấy mode của proxy index trong các cặp hợp lệ.
            3. Dùng proxy mode làm proxy dự đoán.

        Parameters
        ----------
        x:
            [B, N, input_dim]

        mask:
            [B, N], True = padding.

        return_details:
            False:
                Trả 6 output như predict hiện tại.

            True:
                Trả thêm dictionary chứa thông tin top-k và
                proxy dự đoán.

        Returns
        -------
        mu_pred:
            [B, H]

        sigma_pred:
            [B, H]

        mu_proxies:
            [P, H]

        sigma_proxies:
            [P, H]

        features_topk:
            [B, top_k, H]

        logits:
            [P, P]

        details, nếu return_details=True:
            Dictionary chứa index, mask và số phiếu proxy.
        """

        features = self._encode(x)

        B, N, H = features.shape

        padding_mask = self._padding_mask(
            x,
            mask,
        )

        (
            mu_proxies,
            sigma_proxies,
            _,
            mean_proxies,
            logits,
        ) = self._sample_proxies(
            deterministic=True,
        )

        # [B, N, H] @ [H, P] -> [B, N, P]
        similarity = features @ mean_proxies.transpose(0, 1)

        (
            features_topk,
            features_topk_mask,
            topk_token_indices,
            topk_proxy_indices,
            topk_similarity_values,
        ) = self._topk_across_proxies(
            features=features,
            similarity=similarity,
            padding_mask=padding_mask,
        )

        # ---------------------------------------------------------
        # Mode proxy, chỉ tính các cặp hợp lệ.
        # ---------------------------------------------------------
        proxy_counts = torch.zeros(
            B,
            self.num_proxies,
            dtype=torch.long,
            device=features.device,
        )

        # Những vị trí padding/lặp do thiếu cặp có trọng số 0.
        proxy_counts.scatter_add_(
            dim=1,
            index=topk_proxy_indices,
            src=features_topk_mask.long(),
        )

        # Nếu hòa số phiếu, argmax chọn proxy có index nhỏ hơn.
        predicted_proxy_idx = proxy_counts.argmax(dim=1)

        mu_pred = mu_proxies[
            predicted_proxy_idx
        ]

        sigma_pred = sigma_proxies[
            predicted_proxy_idx
        ]

        base_output = (
            mu_pred,
            sigma_pred,
            mu_proxies,
            sigma_proxies,
            features_topk,
            logits,
        )

        if not return_details:
            return base_output

        details = {
            "predicted_proxy_idx": predicted_proxy_idx,      # [B]
            "proxy_counts": proxy_counts,                   # [B, P]
            "topk_token_indices": topk_token_indices,       # [B, K]
            "topk_proxy_indices": topk_proxy_indices,       # [B, K]
            "topk_similarity_values": topk_similarity_values,  # [B, K]
            "features_topk_mask": features_topk_mask,       # [B, K]
        }

        return (*base_output, details)
    def _topk_across_proxies(
        self,
        features,
        similarity,
        padding_mask,
    ):
        """
        Chọn top-k CẶP (feature, proxy) có similarity cao nhất.

        Parameters
        ----------
        features:
            [B, N, H]

        similarity:
            [B, N, P]

        padding_mask:
            [B, N]
            True  = padding
            False = token thật

        Returns
        -------
        selected_features:
            [B, top_k, H]

        selected_mask:
            [B, top_k]
            True  = cặp hợp lệ
            False = vị trí bổ sung do thiếu cặp hợp lệ

        selected_token_idx:
            [B, top_k]

        selected_proxy_idx:
            [B, top_k]

        selected_values:
            [B, top_k]

        Lưu ý
        -----
        Một token có thể được chọn nhiều lần nếu nó nằm trong
        nhiều cặp token–proxy có similarity cao.
        """
        B, N, H = features.shape
        P = similarity.size(-1)

        if similarity.shape != (B, N, self.num_proxies):
            raise ValueError(
                "similarity phải có shape "
                f"{(B, N, self.num_proxies)}, "
                f"nhận được {tuple(similarity.shape)}"
            )

        # Không cho các token padding tham gia lựa chọn.
        masked_similarity = similarity.masked_fill(
            padding_mask.unsqueeze(-1),
            float("-inf"),
        )

        # [B, N, P] -> [B, N*P]
        flat_similarity = masked_similarity.reshape(B, N * P)

        # Không thể lấy nhiều hơn N*P phần tử vật lý.
        K0 = min(self.top_k, N * P)

        top_values, flat_indices = torch.topk(
            flat_similarity,
            k=K0,
            dim=1,
        )

        # flat_index = token_index * P + proxy_index
        token_indices = torch.div(
            flat_indices,
            P,
            rounding_mode="floor",
        )

        proxy_indices = flat_indices % P

        # Kiểm tra cặp được chọn có thuộc token thật không.
        valid = torch.gather(
            ~padding_mask,
            dim=1,
            index=token_indices,
        )

        # _padding_mask đã đảm bảo mỗi bệnh nhân có ít nhất
        # một token thật, nên top-1 là cặp hợp lệ nếu
        # similarity hữu hạn.
        fallback_token = token_indices[:, :1].expand_as(token_indices)
        fallback_proxy = proxy_indices[:, :1].expand_as(proxy_indices)

        safe_token_indices = torch.where(
            valid,
            token_indices,
            fallback_token,
        )

        safe_proxy_indices = torch.where(
            valid,
            proxy_indices,
            fallback_proxy,
        )

        # Khi top_k > N*P, bổ sung các vị trí lặp.
        # Các vị trí này được đánh dấu False trong selected_mask.
        if self.top_k > K0:
            extra_count = self.top_k - K0

            extra_token_indices = token_indices[:, :1].expand(
                B,
                extra_count,
            )

            extra_proxy_indices = proxy_indices[:, :1].expand(
                B,
                extra_count,
            )

            safe_token_indices = torch.cat(
                [safe_token_indices, extra_token_indices],
                dim=1,
            )

            safe_proxy_indices = torch.cat(
                [safe_proxy_indices, extra_proxy_indices],
                dim=1,
            )

            extra_valid = torch.zeros(
                B,
                extra_count,
                dtype=torch.bool,
                device=features.device,
            )

            valid = torch.cat(
                [valid, extra_valid],
                dim=1,
            )

            extra_values = torch.zeros(
                B,
                extra_count,
                dtype=top_values.dtype,
                device=top_values.device,
            )

            top_values = torch.cat(
                [top_values, extra_values],
                dim=1,
            )

        # Không để -inf đi ra các phép tính tiếp theo.
        selected_values = top_values.masked_fill(
            ~valid,
            0.0,
        )

        selected_features = torch.gather(
            features,
            dim=1,
            index=safe_token_indices.unsqueeze(-1).expand(
                -1,
                -1,
                H,
            ),
        )

        return (
            selected_features,
            valid,
            safe_token_indices,
            safe_proxy_indices,
            selected_values,
        )

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
