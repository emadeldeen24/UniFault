import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath

from .Transformer_utils import FullAttention, AttentionLayer, Encoder, EncoderLayer


# ------------ Transformer design ----------------------------------------
class Mlp(L.LightningModule):
    def __init__(self, in_features, mlp_factor=3, drop=0.):
        super().__init__()
        hidden_features = in_features * mlp_factor
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class PatchEmbed(L.LightningModule):
    def __init__(self, seq_len, patch_size=16, stride=16, embed_dim=768):
        super().__init__()

        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches

        self.kernel = patch_size
        self.stride = patch_size
        self.input_layer = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        x = x.unfold(dimension=-1, size=self.kernel, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')
        x_out = self.input_layer(x)
        return x_out


class AttentionBlock(L.LightningModule):
    def __init__(self, embed_dim, mlp_factor, num_heads, dropout=0.0):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(in_features=embed_dim, mlp_factor=mlp_factor, drop=dropout)
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()

    def forward(self, x):
        x = self.layer_norm_1(x)
        x = x + self.attn(x, x, x)[0]
        x = x + self.mlp(self.layer_norm_2(x))
        return x


class Transformer_bkbone(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.patch_embed = PatchEmbed(
            seq_len=args.seq_len, patch_size=args.patch_size, stride=args.patch_size, embed_dim=args.embed_dim
        )

        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, args.embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=args.dropout)

        lora_rank = None
        lora_alpha = 16
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 1, attention_dropout=args.dropout,
                                      output_attention=False), args.embed_dim, args.heads, lora_rank=lora_rank,
                        lora_alpha=lora_alpha),
                    args.embed_dim,
                    4 * args.embed_dim,
                    dropout=args.dropout,
                    activation='gelu'
                ) for _ in range(args.depth)
            ],
            norm_layer=torch.nn.LayerNorm(args.embed_dim)
        )

        self.input_layer = nn.Linear(args.patch_size, args.embed_dim)
        self.pretrain_head = nn.Linear(args.embed_dim, args.patch_size)

        # Classifier head
        self.head = nn.Linear(args.embed_dim, args.num_classes)

    def forward(self, x):
        x_patch = self.patch_embed(x)
        # x: [Batch * Channel, num of Patches, Embed_dim]
        x_patch = x_patch + self.pos_embed
        # x: [Batch * Channel, num of Patches, Embed_dim]
        x_patch = self.pos_drop(x_patch)
        # x: [Batch * Channel, num of Patches, Embed_dim]
        features, _ = self.encoder(x_patch)
        # x: [Batch * Channel, num of Patches, Embed_dim] --> [Batch, Channel * num_patches, Embed_dim]
        features = torch.reshape(features, (-1, self.args.num_channels * features.shape[-2], features.shape[-1]))

        return features

    def predict(self, features):
        features_flat = features.mean(1)
        predictions = self.head(features_flat).squeeze()
        return predictions
