# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from block import blocks, disentangle_decoder, feature_encoder, weibull


# class MultiModelWeibull(nn.Module):
#     def __init__(self, input_dim:int = 128, hidden_dim:int = 128, num_blocks:int = 3, 
#                  dropout:float=0.1, num_proxies:int = 8, top_k:int=256, sample_num:int = 30, seed:int = 42, beta:float = 1.0,
#                  num_head:int = 8, qkv_scale:float = 0.5, qkv_bias:bool = False, droppath_rate:float = 0.1, num_member:int = 3, omic_sizes: list = []):
#         super(MultiModelWeibull, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.num_blocks = num_blocks
#         self.dropout = dropout
#         self.num_proxies = num_proxies
#         self.top_k = top_k
#         self.sample_num = sample_num
#         self.seed = seed
#         self.beta = beta
#         self.num_head = num_head
#         self.qkv_scale = qkv_scale
#         self.qkv_bias = qkv_bias
#         self.droppath_rate = droppath_rate
#         self.num_member = num_member
#         self.omic_sizes = omic_sizes

#         self.Gene_encoder = feature_encoder.SharedMultiGroupEncoder(group_sizes= self.omic_sizes, hidden_dim=self.hidden_dim, num_blocks=self.num_blocks, dropout=self.dropout)

#         self.PIB_histology = feature_encoder.ModalEncoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim, 
#                                                           num_blocks=self.num_blocks, dropout=self.dropout, num_proxies=self.num_proxies, 
#                                                           top_k=self.top_k, sample_num=self.sample_num, seed=self.seed, beta=self.beta)
        
#         self.PIB_pathway = feature_encoder.ModalEncoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim, 
#                                                           num_blocks=self.num_blocks, dropout=self.dropout, num_proxies=self.num_proxies, 
#                                                           top_k=self.top_k, sample_num=self.sample_num, seed=self.seed, beta=self.beta)
        
#         self.PoE = blocks.PoE(modality_num=2, sample_num=self.sample_num, seed=self.seed)
        
#         self.PID = disentangle_decoder.DisentangleDecoder(input_dim = self.hidden_dim, hidden_dim=self.hidden_dim, num_head=self.num_head,
#                                                            qkv_scale=self.qkv_scale, qkv_bias=self.qkv_bias, dropout_rate=self.dropout, 
#                                                            droppath_rate=self.droppath_rate)

#         self.WeibullMix = weibull.WeibullMix(input_dim=self.hidden_dim * 3, hidden_dim=self.hidden_dim, num_member=self.num_member, dropout=self.dropout)

#     def forward(self, histology_input, pathway_input, bins, event_time, censor, mask):
#         gene_features = self.Gene_encoder(histology_input)
#         histology_similarity_positive_mean, histology_similarity_negative_mean, histology_mu_true, histology_sigma_true, histology_mu_proxies, histology_sigma_proxies, histology_features_topk, histology_logits = self.PIB_histology(histology_input, bins, censor, mask)
#         pathway_similarity_positive_mean, pathway_similarity_negative_mean, pathway_mu_true, pathway_sigma_true, pathway_mu_proxies, pathway_sigma_proxies, pathway_features_topk, pathway_logits = self.PIB_pathway(pathway_input, bins, censor, None)
#         mu = torch.stack([histology_mu_true, pathway_mu_true], dim=1)
#         sigma = torch.stack([histology_sigma_true, pathway_sigma_true], dim=1)
#         global_features = self.PoE(mu, sigma)
#         [histology_features, pathway_features, gene_features], _ = self.PID(histology_features, pathway_features, global_features, return_attn=False)
#         combined_features = torch.cat([histology_features, pathway_features, gene_features], dim=-1)
#         shapes, scales, weights = self.WeibullMix(combined_features)
#         return shapes, scales, weights

import torch
from torch import nn
from block import blocks, disentangle_decoder, feature_encoder, weibull


class MultiModelWeibull(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, num_blocks=3,
                 dropout=0.1, num_proxies=8, top_k=256, sample_num=30,
                 seed=42, beta=1.0, num_head=8, qkv_scale=0.5,
                 qkv_bias=False, droppath_rate=0.1, num_member=3, omic_sizes=None):
        super().__init__()
        if num_proxies < 4 or num_proxies % 2:
            raise ValueError('num_proxies must equal 2*n_bins, with n_bins >= 2')
        if top_k <= 0 or hidden_dim % num_head:
            raise ValueError('top_k > 0 and hidden_dim divisible by num_head required')
        self.Gene_encoder = feature_encoder.SharedMultiGroupEncoder(
            omic_sizes or [], hidden_dim, num_blocks, dropout)
        args = dict(hidden_dim=hidden_dim, num_blocks=num_blocks, dropout=dropout,
                    num_proxies=num_proxies, top_k=top_k, sample_num=sample_num,
                    seed=seed, beta=beta)
        self.PIB_histology = feature_encoder.ModalEncoder(input_dim=input_dim, **args)
        self.PIB_pathway = feature_encoder.ModalEncoder(input_dim=hidden_dim, **args)
        # Proxy auxiliary prediction is discrete survival over n_bins, not 2*n_bins.
        self.PIB_histology.decoder = nn.Linear(hidden_dim, num_proxies // 2)
        self.PIB_pathway.decoder = nn.Linear(hidden_dim, num_proxies // 2)
        self.PoE = blocks.PoE(2, sample_num, seed)
        self.PID = disentangle_decoder.DisentangleDecoder(
            hidden_dim, hidden_dim, num_head, qkv_scale, qkv_bias,
            dropout, droppath_rate)
        self.WeibullMix = weibull.WeibullMix(hidden_dim*3, hidden_dim, num_member, dropout)

    @staticmethod
    def _aux(values):
        pos, neg, mu, sigma, all_mu, all_sigma, tokens, logits = values
        return dict(positive=pos, negative=neg, mu=mu, sigma=sigma,
                    all_mu=all_mu, all_sigma=all_sigma, logits=logits)

    def forward(self, histology_input, pathway_input, bins, event_time=None,
                censor=None, mask=None, return_aux=False):
        if censor is None:
            raise ValueError('Training forward requires censor; use predict for inference')
        pathways = self.Gene_encoder(pathway_input)
        h = self.PIB_histology(histology_input, bins, censor, mask)
        p = self.PIB_pathway(pathways, bins, censor)
        common = self.PoE(torch.stack([h[2], p[2]], dim=1),
                          torch.stack([h[3].square(), p[3].square()], dim=1))
        # MIAttention assumes exactly one common token.
        common = common.mean(dim=1, keepdim=True)
        (h_tokens, p_tokens, common_tokens), _ = self.PID(h[6], p[6], common)
        features = (h_tokens.mean(1), p_tokens.mean(1), common_tokens.mean(1))
        x = torch.cat(features, dim=-1)
        shapes, scales, weights = self.WeibullMix(x)
        if not return_aux:
            return shapes, scales, weights
        return dict(x=x, shapes=shapes, scales=scales, weights=weights,
                    histology=self._aux(h), pathway=self._aux(p), features=features)

    def predict(self, histology_input, pathway_input, mask=None):
        # Reuse label-free proxy selection; padding is removed patient by patient.
        if self.training:
            raise RuntimeError('Call model.eval() before predict')
        if mask is not None:
            results = []
            for i in range(histology_input.size(0)):
                valid = ~mask[i].bool()
                if not valid.any():
                    raise ValueError('No valid histology token')
                results.append(self.predict(histology_input[i:i+1, valid],
                                           [g[i:i+1] for g in pathway_input]))
            return {key: torch.cat([r[key] for r in results], dim=0) for key in results[0]}
        pathways = self.Gene_encoder(pathway_input)
        h = self.PIB_histology.predict(histology_input)
        p = self.PIB_pathway.predict(pathways)
        common = self.PoE(torch.stack([h[0], p[0]], 1),
                          torch.stack([h[1].square(), p[1].square()], 1)).mean(1, keepdim=True)
        tokens, _ = self.PID(h[4], p[4], common)
        x = torch.cat([v.mean(1) for v in tokens], -1)
        return self.WeibullMix.predict_expectation(x)