from math import sqrt
import torch
import torch.nn as nn


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class LoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank, alpha):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.lora_A = nn.Parameter(torch.randn(output_dim, rank))
        self.lora_B = nn.Parameter(torch.randn(rank, input_dim))

    def forward(self, x):
        # Low-rank adaptation
        lora_delta = self.alpha * self.lora_A @ self.lora_B
        return x @ lora_delta.transpose(-1, -2)


class LoRAPrarametrization(nn.Module):
    def __init__(self, features_in, features_out, rank, alpha, device='cpu'):
        super().__init__()
        # We use a random Gaussian initialization for A and zero for B, so W = BA is zero at the beginning of the training
        self.lora_A = nn.Parameter(torch.zeros((rank, features_out)).to(device))
        self.lora_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))
        nn.init.normal_(self.lora_A, mean=0, std=1)
        # This scale helps to reduce the need to retune hyperparameters when we vary r
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, original_weights):
        if self.enabled:
            # Return W + (B*A)*scale
            return original_weights + torch.matmul(self.lora_B, self.lora_A).view(original_weights.shape) * self.scale
        else:
            return original_weights


import torch.nn.utils.parametrize as parametrize


def linear_layer_parametrization(layer, device, rank, lora_alpha):
    features_in, features_out = layer.weight.shape
    return LoRAPrarametrization(
        features_in, features_out, rank=rank, alpha=lora_alpha, device=device
    )


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, lora_rank=None, lora_alpha=4):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.lora_rank = lora_rank

        if lora_rank is not None:
            parametrize.register_parametrization(
                self.query_projection,
                "weight",
                linear_layer_parametrization(self.query_projection, device=self.query_projection.weight.device,
                                             rank=lora_rank, lora_alpha=lora_alpha)
            )
            parametrize.register_parametrization(
                self.key_projection,
                "weight", linear_layer_parametrization(self.key_projection, device=self.key_projection.weight.device,
                                                       rank=lora_rank, lora_alpha=lora_alpha)
            )
            parametrize.register_parametrization(
                self.value_projection,
                "weight",
                linear_layer_parametrization(self.value_projection, device=self.value_projection.weight.device,
                                             rank=lora_rank, lora_alpha=lora_alpha)
            )

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Linear(d_model, d_ff)
        self.conv2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)

        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)

        x = self.norm(x)

        return x, attns
