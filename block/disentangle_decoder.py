import torch
import torch.nn as nn
from . import blocks

class DisentangleDecoder(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, num_head:int, qkv_scale:float, qkv_bias:bool = False,dropout_rate:float=0.1, droppath_rate:float = 0.1):
        super(DisentangleDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.qkv_scale = qkv_scale
        self.qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate
        self.droppath_rate = droppath_rate
        self.mlp = blocks.MLP(input_feature=self.input_dim, hidden_feature= self.hidden_dim, dropout_rate= self.dropout_rate)
        self.attention = blocks.MIAttention(dim= self.input_dim, num_heads= self.num_head, qkv_bias= self.qkv_bias, qk_scale= self.qkv_scale, attn_drop=self.dropout_rate, proj_drop= self.dropout_rate)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(input_dim)
        self.droppath = blocks.DropPath(drop_prob= self.droppath_rate)
    def forward(self, histology_feature, pathway_feature, global_feature, return_attn:bool = False):
        [attn_histology_x, attn_pathways_x, attn_global_x], attn = self.attention(self.norm(histology_feature), self.norm(pathway_feature), self.norm(global_feature), return_attn)
        histology_feature = histology_feature + self.droppath(attn_histology_x)
        pathway_feature = pathway_feature + self.droppath(attn_pathways_x)
        global_feature = global_feature + self.droppath(attn_global_x)
        histology_feature = histology_feature + self.mlp(self.norm(histology_feature))
        pathway_feature = pathway_feature + self.mlp(self.norm(pathway_feature))
        global_feature = global_feature + self.mlp(self.norm(global_feature))
        if return_attn:
            return [attn_histology_x, attn_pathways_x, attn_global_x], attn
        else:
                return [attn_histology_x, attn_pathways_x, attn_global_x], None