import torch
import torch.nn as nn
import torch.nn.functional as F

def drop_path_fast(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
        
    keep_prob = 1 - drop_prob
    
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    
    random_tensor = torch.bernoulli(torch.full(shape, keep_prob, dtype=x.dtype, device=x.device))
    
    return x * (random_tensor / keep_prob)


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_fast(x, self.drop_prob, self.training)

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, drop_path_prob: float = 0.0):
        super(ResidualBlock, self).__init__()
        # Giữ nguyên các tầng tính toán của em
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Tích hợp thêm DropPath tối ưu (Stochastic Depth) cho nhánh chính
        # Nếu drop_path_prob = 0, nó sẽ tự động trở thành một tầng Identity (giữ nguyên dữ liệu)
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()
        self.dim = dim

    def forward(self, x):
        out = self.norm(x)
        out = self.linear(out)
        out = self.dropout(self.activate(out))
    
        return x + self.drop_path(out)
    
class Proxies(nn.Module):
    def __init__(self, num_proxies: int, dim: int):
        super(Proxies, self).__init__()
        self.num_proxies = num_proxies 
        self.dim = dim
        
        # Ma trận chứa các tham số proxy học được
        self.proxies = nn.Parameter(torch.randn(num_proxies, dim * 2))
        
        # ---- TỐI ƯU: TẠO PROXY LOOKUP TENSOR ----
        # Số lượng class 'c' tối đa là num_proxies // 2
        num_c = num_proxies // 2
        num_y = 2  # y luôn là 0 hoặc 1
        
        # Tạo sẵn một ma trận ánh xạ kích thước [num_c, num_y]
        lookup_table = torch.zeros((num_c, num_y), dtype=torch.long)
        for i in range(num_c):
            for j in range(num_y):
                lookup_table[i, j] = i * 2 + j
                
        # Đăng ký lookup_table dưới dạng buffer để PyTorch tự quản lý thiết bị (CPU/GPU)
        self.register_buffer('proxy_lookup', lookup_table)
        # ----------------------------------------

    def forward(self):
        mu_proxies = self.proxies[:, :self.dim]
        # Dùng softplus để đảm bảo độ lệch chuẩn sigma luôn dương
        sigma_proxies = nn.functional.softplus(self.proxies[:, self.dim:])
        return mu_proxies, sigma_proxies
    
class MIAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.,
    ):
        super(MIAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, histology, pathways, global_feature, return_attn=False):
        B, N_histology, C = histology.shape
        _, N_pathways, _ = pathways.shape

        # --- BƯỚC 1: TỐI ƯU HÓA QKV GOM CỤM (CONCATENATION TRƯỚC KHI CHIẾU) ---
        # Thay vì gọi self.qkv 3 lần, ta gom toàn bộ chuỗi token của 3 phương thức lại làm một.
        # Tổng số lượng tokens: 1 (global) + N_histology + N_pathways
        # Shape sau concat: [B, 1 + N_histology + N_pathways, C]
        all_features = torch.cat([global_feature, histology, pathways], dim=1)
        
        # Chỉ gọi một phép tính tuyến tính duy nhất cho toàn bộ dữ liệu. Tối ưu cực lớn cho CUDA GPU!
        # Shape sau khi biến đổi: [3, B, num_heads, Total_Tokens, head_dim]
        qkv_all = (self.qkv(all_features)
                   .reshape(B, -1, 3, self.num_heads, C // self.num_heads)
                   .permute(2, 0, 3, 1, 4))
        
        q_all, k_all, v_all = qkv_all[0], qkv_all[1], qkv_all[2]

        # Phân rã ma trận Q, K, V quay trở lại các phương thức ban đầu bằng phép cắt ma trận (Slicing)
        # Cách này nhanh hơn việc unbind hoặc gọi rời rạc rất nhiều vì dữ liệu đã nằm liên tục trên RAM.
        q_global    = q_all[:, :, 0:1, :]
        q_histology = q_all[:, :, 1:1+N_histology, :]
        q_pathways  = q_all[:, :, 1+N_histology:, :]

        k_histology = k_all[:, :, 1:1+N_histology, :]
        k_pathways  = k_all[:, :, 1+N_histology:, :]

        v_histology = v_all[:, :, 1:1+N_histology, :]
        v_pathways  = v_all[:, :, 1+N_histology:, :]

        # --- BƯỚC 2: TÍNH TOÁN MA TRẬN ATTENTION SONG SONG ---
        # Self-attention nội bộ phương thức
        attn_histology = (q_histology @ k_histology.transpose(-2, -1)) * self.scale
        attn_pathways = (q_pathways @ k_pathways.transpose(-2, -1)) * self.scale

        # Cross-attention cho tính chất toàn cục (Global sẽ nhìn vào toàn bộ không gian k_all)
        # Code cũ: torch.cat((k_global, k_histology, k_pathways)) tốn bộ nhớ sao chép dữ liệu.
        # Đã tối ưu: Dùng luôn ma trận k_all tổng thể ban đầu mà không cần concat lại!
        attn_global = (q_global  @ k_all.transpose(-2, -1)) * self.scale

        # Áp dụng Softmax và Dropout chuẩn hóa
        attn_histology = self.attn_drop(attn_histology.softmax(dim=-1))
        attn_pathways = self.attn_drop(attn_pathways.softmax(dim=-1))
        attn_global = self.attn_drop(attn_global.softmax(dim=-1))

        # --- BƯỚC 3: TỔNG HỢP GIÁ TRỊ VÀ KẾT HỢP DỮ LIỆU ĐẦU RA ---
        attn_histology_x = (attn_histology @ v_histology).transpose(1, 2).reshape(B, N_histology, C)
        attn_pathways_x = (attn_pathways @ v_pathways).transpose(1, 2).reshape(B, N_pathways, C)
        
        # Tương tự như bước trên, dùng trực tiếp v_all thay vì gọi phép toán nối torch.cat một lần nữa
        attn_global_x = (attn_global @ v_all).transpose(1, 2).reshape(B, 1, C)

        # Gom các kết quả đầu ra đi qua tầng Linear cuối và Dropout song song
        # Thay vì gọi self.proj và self.proj_drop 3 lần độc lập, ta gom chúng lại xử lý một lượt
        out_concat = torch.cat([attn_global_x, attn_histology_x, attn_pathways_x], dim=1)
        out_projected = self.proj_drop(self.proj(out_concat))

        # Cắt ma trận để trả lại đúng 3 danh sách đặc trưng ban đầu
        attn_global_x = out_projected[:, 0:1, :]
        attn_histology_x = out_projected[:, 1:1+N_histology, :]
        attn_pathways_x = out_projected[:, 1+N_histology:, :]

        if return_attn:
            return [attn_histology_x, attn_pathways_x, attn_global_x], [attn_histology, attn_pathways]
        else:
            return [attn_histology_x, attn_pathways_x, attn_global_x], None
        

# class PoE(nn.Module):
#     def __init__(self, modality_num=2, sample_num=50, seed=1):
#         super(PoE, self).__init__()
#         self.sample_num = sample_num
#         self.seed = seed
#         self.phi = nn.Parameter(torch.ones(modality_num))

#     def forward(self, mu_list, var_list, eps_val=1e-8):
#         """
#         mu_list, var_list: List các Tensor dạng [Batch, Dim]
#         """
#         # 1. Chuyển list sang Tensor để tính toán song song [Modality, Batch, Dim]
#         mu_stack = torch.stack(mu_list) 
#         var_stack = torch.stack(var_list)
        
#         # 2. Tính toán Alpha động qua Softmax
#         alpha = F.softmax(self.phi, dim=0) # [Modality]
#         # Biến alpha thành dạng [Modality, 1, 1] để tự động nhân bản (broadcast)
#         alpha = alpha.view(-1, 1, 1)

#         # 3. Tính toán Precision (T) và hợp nhất phân phối (Vectorized)
#         precision_stack = 1.0 / (var_stack + eps_val)
        
#         # Tổng Precision có trọng số alpha: 1 / var_joint
#         t_sum = torch.sum(alpha * precision_stack, dim=0)
#         var_joint = 1.0 / (t_sum + eps_val)
        
#         # Trung bình chung có trọng số: (Sum alpha * T * mu) / t_sum
#         mu_joint = torch.sum(alpha * precision_stack * mu_stack, dim=0) * var_joint

#         # 4. Lấy mẫu đặc trưng (Reparameterization Trick)
#         # Z = mu + sqrt(var) * epsilon
#         std_joint = torch.sqrt(var_joint)
        
#         batch_size, dim = mu_joint.shape
#         noise = self.gaussian_noise(batch_size, self.sample_num, dim)
        
#         # Tăng chiều để nhân với noise: [B, 1, D] + [B, 1, D] * [B, S, D]
#         poe_features = mu_joint.unsqueeze(1) + std_joint.unsqueeze(1) * noise
        
#         return poe_features

#     def gaussian_noise(self, batch_size, sample_num, dim):
#         shape = (batch_size, sample_num, dim)
#         device = self.phi.device # Tự động lấy device của mô hình
        
#         if self.training:
#             return torch.randn(shape, device=device)
#         else:
#             # Cố định seed khi đánh giá để kết quả không thay đổi
#             gen = torch.Generator(device=device).manual_seed(self.seed)
#             return torch.randn(shape, device=device, generator=gen)




class MLP(nn.Module):
    def __init__(self, input_feature:int, hidden_feature:int, dropout_rate:float, output_feature:int = None):
        super(MLP, self).__init__()
        self.input_feature = input_feature
        self.hidden_feature = hidden_feature
        self.dropout_rate = dropout_rate
        self.output_feature = output_feature
        self.linear1 = nn.Linear(in_features= self.input_feature, out_features= self.hidden_feature)
        self.linear2 = nn.Linear(in_features= self.hidden_feature, out_features= self.output_feature if self.output_feature is not None else self.input_feature)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p = self.dropout_rate)
    def forward(self,x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorOrList = Union[
    torch.Tensor,
    Sequence[torch.Tensor],
]


class PoE(nn.Module):
    """
    Weighted Product of Gaussian Experts.

    Input:
        mu  : [B, M, D]
        var : [B, M, D]

    Output:
        samples   : [B, S, D]
        mu_joint  : [B, D]
        var_joint : [B, D]
        alpha     : [M]
    """

    def __init__(
        self,
        modality_num: int = 2,
        sample_num: int = 50,
        seed: int = 1,
        eps: float = 1e-8,
    ):
        super().__init__()

        if modality_num <= 0:
            raise ValueError(
                "modality_num phải lớn hơn 0"
            )

        if sample_num <= 0:
            raise ValueError(
                "sample_num phải lớn hơn 0"
            )

        self.modality_num = modality_num
        self.sample_num = sample_num
        self.seed = seed
        self.eps = eps

        # Softmax(phi) ban đầu cho trọng số đều
        self.phi = nn.Parameter(
            torch.zeros(modality_num)
        )

    def forward(
        self,
        mu: TensorOrList,
        var: TensorOrList,
        return_distribution: bool = False,
    ):
        # Chuẩn hóa đầu vào thành [B, M, D]
        mu = self._to_bmd_tensor(mu, name="mu")
        var = self._to_bmd_tensor(var, name="var")

        self._validate_shapes(mu, var)

        original_dtype = mu.dtype
        device = mu.device

        # PoE nhạy với phép nghịch đảo.
        # Nên tính bằng float32 khi đầu vào là FP16/BFloat16.
        compute_dtype = (
            torch.float32
            if original_dtype in {
                torch.float16,
                torch.bfloat16,
            }
            else original_dtype
        )

        mu = mu.to(dtype=compute_dtype)
        var = var.to(dtype=compute_dtype)

        # var phải dương
        var = var.clamp_min(self.eps)

        # alpha: [M]
        alpha = F.softmax(
            self.phi,
            dim=0,
        ).to(
            device=device,
            dtype=compute_dtype,
        )

        # [M] → [1, M, 1]
        alpha_broadcast = alpha.view(
            1,
            self.modality_num,
            1,
        )

        # precision của từng modality:
        #
        # precision_m = alpha_m / var_m
        #
        # Shape: [B, M, D]
        weighted_precision = (
            alpha_broadcast * torch.reciprocal(var)
        )

        # Cộng theo chiều modality
        #
        # [B, M, D] → [B, D]
        joint_precision = weighted_precision.sum(
            dim=1
        )

        joint_precision = joint_precision.clamp_min(
            self.eps
        )

        # var_joint = 1 / joint_precision
        #
        # Shape: [B, D]
        var_joint = torch.reciprocal(
            joint_precision
        )

        # mu_joint =
        # sum_m(precision_m * mu_m)
        # --------------------------------
        # sum_m(precision_m)
        #
        # Shape: [B, D]
        mu_joint = (
            weighted_precision * mu
        ).sum(dim=1) * var_joint

        # Tương đương:
        #
        # mu_joint = (
        #     (weighted_precision * mu).sum(dim=1)
        #     / joint_precision
        # )

        std_joint = torch.sqrt(
            var_joint.clamp_min(self.eps)
        )

        batch_size, latent_dim = mu_joint.shape

        # noise: [B, S, D]
        noise = self._gaussian_noise(
            batch_size=batch_size,
            sample_num=self.sample_num,
            latent_dim=latent_dim,
            device=device,
            dtype=compute_dtype,
        )

        # [B, D] → [B, 1, D]
        #
        # [B, 1, D] + [B, 1, D] * [B, S, D]
        # → [B, S, D]
        samples = (
            mu_joint.unsqueeze(1)
            + std_joint.unsqueeze(1) * noise
        )

        samples = samples.to(original_dtype)
        mu_joint = mu_joint.to(original_dtype)
        var_joint = var_joint.to(original_dtype)
        alpha = alpha.to(original_dtype)

        if return_distribution:
            return (
                samples,
                mu_joint,
                var_joint,
                alpha,
            )

        return samples

    def _to_bmd_tensor(
        self,
        value: TensorOrList,
        name: str,
    ) -> torch.Tensor:
        """
        Chuyển đầu vào thành [B, M, D].

        Tensor:
            Đầu vào phải là [B, M, D].

        List:
            M phần tử [B, D]
            được stack theo dim=1 thành [B, M, D].
        """

        if isinstance(value, torch.Tensor):
            if value.ndim != 3:
                raise ValueError(
                    f"{name} phải có shape [B, M, D], "
                    f"nhận được {tuple(value.shape)}"
                )

            return value

        if len(value) != self.modality_num:
            raise ValueError(
                f"{name} cần {self.modality_num} modality, "
                f"nhận được {len(value)}"
            )

        # M tensor [B, D] → [B, M, D]
        return torch.stack(
            tuple(value),
            dim=1,
        )

    def _validate_shapes(
        self,
        mu: torch.Tensor,
        var: torch.Tensor,
    ) -> None:
        if mu.shape != var.shape:
            raise ValueError(
                f"mu và var phải cùng shape, "
                f"nhận được mu={tuple(mu.shape)}, "
                f"var={tuple(var.shape)}"
            )

        if mu.ndim != 3:
            raise ValueError(
                f"mu và var phải là [B, M, D], "
                f"nhận được {tuple(mu.shape)}"
            )

        if mu.size(1) != self.modality_num:
            raise ValueError(
                f"Chiều modality phải bằng "
                f"{self.modality_num}, "
                f"nhận được {mu.size(1)}"
            )

        if mu.device != var.device:
            raise ValueError(
                f"mu và var phải cùng device, "
                f"nhận được {mu.device} và {var.device}"
            )

    def _gaussian_noise(
        self,
        batch_size: int,
        sample_num: int,
        latent_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        shape = (
            batch_size,
            sample_num,
            latent_dim,
        )

        if self.training:
            return torch.randn(
                shape,
                device=device,
                dtype=dtype,
            )

        generator = torch.Generator(
            device=device
        )
        generator.manual_seed(self.seed)

        return torch.randn(
            shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )