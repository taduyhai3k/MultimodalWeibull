import torch
import torch.nn as nn
import torch.nn.functional as f 
from . import blocks as b
class WeibullMix(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, num_member:int, dropout:float=0.1, eps:float=1e-7):
        super(WeibullMix, self).__init__()
        self.encoder = b.MLP(input_feature=input_dim, hidden_feature=hidden_dim, dropout_rate=dropout, output_feature=hidden_dim)
        self.num_member = num_member
        self.decoder_shape = b.MLP(input_feature= hidden_dim, hidden_feature= self.num_member * input_dim, dropout_rate=dropout, output_feature= self.num_member * input_dim)
        self.decoder_scale = b.MLP(input_feature= hidden_dim, hidden_feature= self.num_member * input_dim, dropout_rate=dropout, output_feature= self.num_member * input_dim)
        self.decoder_weight = b.MLP(input_feature= hidden_dim, hidden_feature= self.num_member, dropout_rate=dropout, output_feature= self.num_member)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim    
        self.dropout = dropout
        self.eps = eps
    def forward(self, x):
        batch_size = x.size(0)
        encoded = self.encoder(x)
        shapes = self.decoder_shape(encoded).view(batch_size, self.num_member, self.input_dim)
        scales = self.decoder_scale(encoded).view(batch_size, self.num_member, self.input_dim)
        weights = f.softmax(self.decoder_weight(encoded), dim=-1)
        return shapes, scales, weights
    def predict_expectation(self, x):
        with torch.no_grad():
            shape_coeffs, scale_coeffs, weights = self.forward(x)

            shape_logits = torch.einsum(
                "bd,bmd->bm",
                x,
                shape_coeffs
            )

            scale_logits = torch.einsum(
                "bd,bmd->bm",
                x,
                scale_coeffs
            )

            shapes = f.softplus(shape_logits) + self.eps
            scales = f.softplus(scale_logits) + self.eps
            log_component_expectations = (
                torch.lgamma(1.0 + 1.0 / shapes)
                - torch.log(scales) / shapes
            )

            component_expectations = torch.exp(
                log_component_expectations
            )

            prediction = torch.sum(
                weights * component_expectations,
                dim=-1
            )

            return {
                "prediction": prediction,
                "component_expectations": component_expectations,
                "shapes": shapes,
                "scales": scales,
                "weights": weights
            }