"""
SegFormer3D Variant for Pancreatic Tumor Segmentation
Modified from the original SegFormer3D (GPLv3) by the SegFormer3D authors.
Modifications by Kyriaki Kolpetinou, 2025-09-11.
-----------------------------------------------------

Base architecture:
- SegFormer3D encoder (Perera et al., 2023)

Decoder design:
- FPN-style top-down pathway with lateral 1x1s (Lin et al., CVPR 2017)
- ASPP bottleneck (Chen et al., DeepLabV3, 2017)
- scSE modules in skip connections (Roy et al., MICCAI 2018)
- Attention Gates for skip filtering (Oktay et al., arXiv 2018)
- Inspired also by Deng & Mou (ISAIM 2023) for combining ASPP + AG in pancreatic tumor segmentation

This combination is novel in 3D transformer-based segmentation.
Copyright (C) 2025 Kyriaki Kolpetinou
"""

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- scSE: channel-SE (cSE) + spatial-SE (sSE) ---

class cSE3D(nn.Module):
    """Channel squeeze/excitation: GAP -> 1x1 conv bottleneck -> 1x1 conv -> sigmoid -> channel reweight."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x):
        w = self.avg(x)           # (B,C,1,1,1)
        w = self.act(self.fc1(w)) # (B,hidden,1,1,1)
        w = torch.sigmoid(self.fc2(w))  # (B,C,1,1,1)
        return x * w

class sSE3D(nn.Module):
    """Spatial squeeze/excitation: 1x1x1 conv to single-channel map -> sigmoid -> spatial reweight."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, 1, kernel_size=1, bias=True)

    def forward(self, x):
        w = torch.sigmoid(self.conv(x))  # (B,1,D,H,W)
        return x * w

class scSE3D(nn.Module):
    """Concurrent spatial & channel SE: x_cse + x_sse (the original paper sums the two branches)."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.cse = cSE3D(channels, reduction=reduction)
        self.sse = sSE3D(channels)

    def forward(self, x):
        return self.cse(x) + self.sse(x)

# --- 
class DWSeparable3d(nn.Module):
    def __init__(self, c, k=3, d=1):
        super().__init__()
        p = d * (k // 2)
        self.dw = nn.Conv3d(c, c, k, padding=p, dilation=d, groups=c, bias=False)
        self.pw = nn.Conv3d(c, c, 1, bias=False)
        self.norm = nn.InstanceNorm3d(c)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.dw(x); x = self.pw(x); x = self.norm(x); return self.act(x)

def upsample_like(x, ref):
    return F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)

# ---------- Attention Gate (AG) ----------
class AttentionGate3D(nn.Module):
    """
    Gating: uses decoder feature g to modulate skip x.
    Produces a spatial mask psi in [0,1] and gates the skip: x * psi.
    """
    def __init__(self, in_x, in_g, inter_channels):
        super().__init__()
        self.theta_x = nn.Conv3d(in_x, inter_channels, kernel_size=1, bias=False)
        self.phi_g   = nn.Conv3d(in_g, inter_channels, kernel_size=1, bias=False)
        self.act     = nn.GELU()
        self.psi     = nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True)
    def forward(self, x, g):
        # match spatial size
        if x.shape[2:] != g.shape[2:]:
            g = upsample_like(g, x)
        theta = self.theta_x(x)
        phi   = self.phi_g(g)
        a = self.act(theta + phi)
        psi = torch.sigmoid(self.psi(a))   # (B,1,D,H,W)
        return x * psi

# ---------- ASPP3D (lightweight) ----------
class ASPP3D(nn.Module):
    """
    Parallel branches: 1x1x1 + dilated depthwise (rates) + image pooling.
    Projects to out_ch at the end.
    """
    def __init__(self, in_ch, out_ch, rates=(1,2,3), use_image_pool=True):
        super().__init__()
        self.branches = nn.ModuleList()
        # 1x1x1
        self.branches.append(nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.GELU()
        ))
        # dilated branches
        for r in rates:
            self.branches.append(nn.Sequential(
                DWSeparable3d(in_ch, k=3, d=r),
                nn.Conv3d(in_ch, out_ch, 1, bias=False),
                nn.InstanceNorm3d(out_ch),
                nn.GELU()
            ))
        self.use_pool = use_image_pool
        if use_image_pool:
            self.pool_proj = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(in_ch, out_ch, 1, bias=False),
                nn.GELU()
            )
        n_concat = len(self.branches) + (1 if use_image_pool else 0)
        self.proj = nn.Conv3d(out_ch * n_concat, out_ch, 1, bias=False)

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        if self.use_pool:
            p = self.pool_proj(x)
            p = upsample_like(p, x)
            feats.append(p)
        y = torch.cat(feats, dim=1)
        return self.proj(y)  # (B, out_ch, d,h,w)

# ---------- ASPP+FPN decoder ----------
class ASPPFPNDecoderHead(nn.Module):
    """
    Inputs: c1 (1/4), c2 (1/8), c3 (1/16), c4 (1/32) from encoder.
    Steps:
      - ASPP on c4 -> P4 (C)
      - For k in {3,2,1}:
          skip_k = scSE3D(c{k})
          skip_k = AG(skip_k, current_P)
          Pk = Smooth( up(current_P) + Lateral(skip_k) )
      - Fuse {P4,P3,P2,P1} (upsampled to P1) -> 1x1x1 classifier
    """
    def __init__(self, in_channels, num_classes, C=128, aspp_rates=(1,2,3), fuse_all=True):
        """
        in_channels: list [c1_ch, c2_ch, c3_ch, c4_ch]
        C: decoder channel budget
        fuse_all: if True, concat {P4..P1}; else predict from P1 only
        """
        super().__init__()
        c1_ch, c2_ch, c3_ch, c4_ch = in_channels

        # bottleneck ASPP
        self.aspp = ASPP3D(c4_ch, C, rates=aspp_rates, use_image_pool=True)

        # With scSE
        self.cam1 = scSE3D(c1_ch, reduction=8)
        self.cam2 = scSE3D(c2_ch, reduction=8)
        self.cam3 = scSE3D(c3_ch, reduction=8)

        # Attention gates (x from skip, g from current P)
        self.ag1 = AttentionGate3D(in_x=c1_ch, in_g=C, inter_channels=max(C//2, 32))
        self.ag2 = AttentionGate3D(in_x=c2_ch, in_g=C, inter_channels=max(C//2, 32))
        self.ag3 = AttentionGate3D(in_x=c3_ch, in_g=C, inter_channels=max(C//2, 32))

        # Lateral 1x1 to common C
        self.lat1 = nn.Conv3d(c1_ch, C, 1, bias=False)
        self.lat2 = nn.Conv3d(c2_ch, C, 1, bias=False)
        self.lat3 = nn.Conv3d(c3_ch, C, 1, bias=False)

        # Smoothing convs
        self.smooth3 = DWSeparable3d(C, k=3, d=1)
        self.smooth2 = DWSeparable3d(C, k=3, d=1)
        self.smooth1 = DWSeparable3d(C, k=3, d=1)

        self.fuse_all = fuse_all
        if fuse_all:
            self.fuse = nn.Sequential(
                nn.Conv3d(C*4, C, 1, bias=False),
                nn.InstanceNorm3d(C),
                nn.GELU()
            )
            self.cls = nn.Conv3d(C, num_classes, 1)
        else:
            self.cls = nn.Conv3d(C, num_classes, 1)

        # NEW: auxiliary head (predict from P2; light & stable)
        self.cls_aux = nn.Conv3d(C, num_classes, 1)

    def forward(self, c1, c2, c3, c4):
        # bottleneck
        P4 = self.aspp(c4)  # (B,C, d4,h4,w4)

        # level 3
        x3 = self.cam3(c3)
        x3 = self.ag3(x3, P4)
        P3 = upsample_like(P4, x3) + self.lat3(x3)
        P3 = self.smooth3(P3)

        # level 2
        x2 = self.cam2(c2)
        x2 = self.ag2(x2, P3)
        P2 = upsample_like(P3, x2) + self.lat2(x2)
        P2 = self.smooth2(P2)

        # level 1
        x1 = self.cam1(c1)
        x1 = self.ag1(x1, P2)
        P1 = upsample_like(P2, x1) + self.lat1(x1)
        P1 = self.smooth1(P1)

        if self.fuse_all:
            # upsample all to P1 size and concatenate
            P4u = upsample_like(P4, P1)
            P3u = upsample_like(P3, P1)
            P2u = upsample_like(P2, P1)
            fused = torch.cat([P4u, P3u, P2u, P1], dim=1)
            fused = self.fuse(fused)
            logits_main = self.cls(fused)
        else:
            logits_main = self.cls(P1)

        # NEW: aux from P2 (no upsample here; do it in wrapper)
        logits_aux = self.cls_aux(P2)
        return logits_main, logits_aux
    

# ---------------- Hybrid Stem ----------------
class HybridConvStem3D(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(embed_dim // 2),
            nn.GELU(),
            nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(embed_dim),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.stem(x)  # (B, C, D, H, W)
        B, C, D, H, W = x.shape
        spatial_shape = (D, H, W)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.norm(x)
        return x, spatial_shape



# ---------------- Patch Embedding ----------------
class PatchEmbedding(nn.Module):
    def __init__(self, in_channel, embed_dim, kernel_size, stride, padding):
        super().__init__()
        self.patch_embeddings = nn.Conv3d(
            in_channel, embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embeddings(x)  # (B, C, D, H, W)
        spatial_shape = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.norm(x)
        return x, spatial_shape


# ---------------- DWConv + MLP ----------------
class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, groups=dim)
        self.bn = nn.BatchNorm3d(dim)

    def forward(self, x, spatial_shape):
        B, N, C = x.shape
        D, H, W = spatial_shape
        x = x.transpose(1, 2).view(B, C, D, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, mlp_ratio=4, dropout=0.0):
        super().__init__()
        hidden_dim = int(in_features * mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.dwconv = DWConv(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, spatial_shape):
        x = self.fc1(x)
        x = self.dwconv(x, spatial_shape)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---------------- Self-Attention ----------------
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, sr_ratio=1, qkv_bias=False, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_dropout)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, spatial_shape):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            D, H, W = spatial_shape
            x_ = x.transpose(1, 2).reshape(B, C, D, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).transpose(1, 2)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]  # (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out



class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        mlp_ratio: int = 4,
        num_heads: int = 8,
        sr_ratio: int = 1,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout=proj_dropout)

    def forward(self, x, spatial_shape):
        x = x + self.attention(self.norm1(x), spatial_shape)
        x = x + self.mlp(self.norm2(x), spatial_shape)
        return x


# ---------------- MixVisionTransformer (Encoder) ----------------
class MixVisionTransformer(nn.Module):
    def __init__(self, in_channels, embed_dims, num_heads, depths, sr_ratios, patch_kernel_size, patch_stride, patch_padding, mlp_ratios,use_hybrid_stem=True):
        super().__init__()
        self.use_hybrid_stem = use_hybrid_stem
        if use_hybrid_stem:
            self.embed_1 = HybridConvStem3D(in_channels, embed_dims[0])
        else:
            self.embed_1 = PatchEmbedding(in_channels, embed_dims[0], patch_kernel_size[0], patch_stride[0], patch_padding[0])

        self.embed_2 = PatchEmbedding(embed_dims[0], embed_dims[1], patch_kernel_size[1], patch_stride[1], patch_padding[1])
        self.embed_3 = PatchEmbedding(embed_dims[1], embed_dims[2], patch_kernel_size[2], patch_stride[2], patch_padding[2])
        self.embed_4 = PatchEmbedding(embed_dims[2], embed_dims[3], patch_kernel_size[3], patch_stride[3], patch_padding[3])

        self.tf_block1 = nn.ModuleList([TransformerBlock(embed_dim=embed_dims[0], num_heads=num_heads[0], sr_ratio=sr_ratios[0], mlp_ratio=mlp_ratios[0], qkv_bias=True) for _ in range(depths[0])])
        self.tf_block2 = nn.ModuleList([TransformerBlock(embed_dim=embed_dims[1], num_heads=num_heads[1], sr_ratio=sr_ratios[1], mlp_ratio=mlp_ratios[1], qkv_bias=True) for _ in range(depths[1])])
        self.tf_block3 = nn.ModuleList([TransformerBlock(embed_dim=embed_dims[2], num_heads=num_heads[2], sr_ratio=sr_ratios[2], mlp_ratio=mlp_ratios[2], qkv_bias=True) for _ in range(depths[2])])
        self.tf_block4 = nn.ModuleList([TransformerBlock(embed_dim=embed_dims[3], num_heads=num_heads[3], sr_ratio=sr_ratios[3], mlp_ratio=mlp_ratios[3], qkv_bias=True) for _ in range(depths[3])])

        self.norm1 = nn.LayerNorm(embed_dims[0])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x):
        out = []

        # Stage 1
        x, s1 = self.embed_1(x)
        for blk in self.tf_block1: x = blk(x, s1)
        x = self.norm1(x)
        x = x.transpose(1, 2).view(x.size(0), -1, *s1)
        out.append(x)

        # Stage 2
        x, s2 = self.embed_2(x)
        for blk in self.tf_block2: x = blk(x, s2)
        x = self.norm2(x)
        x = x.transpose(1, 2).view(x.size(0), -1, *s2)
        out.append(x)

        # Stage 3
        x, s3 = self.embed_3(x)
        for blk in self.tf_block3: x = blk(x, s3)
        x = self.norm3(x)
        x = x.transpose(1, 2).view(x.size(0), -1, *s3)
        out.append(x)

        # Stage 4
        x, s4 = self.embed_4(x)
        for blk in self.tf_block4: x = blk(x, s4)
        x = self.norm4(x)
        x = x.transpose(1, 2).view(x.size(0), -1, *s4)
        out.append(x)

        return out


# ---------------- SegFormer3D Wrapper ----------------
class SegFormer3D(nn.Module):

  """
    SegFormer3D Variant for pancreas & tumor segmentation.

    Args:
        in_channels (int): number of input channels (default: 1 for MRI/CT).
        num_classes (int): number of output segmentation classes.
        embed_dims (list[int]): embedding dims per encoder stage.
        depths (list[int]): number of transformer blocks per stage.
        num_heads (list[int]): attention heads per stage.
        sr_ratios (list[int]): spatial reduction ratios.
        patch_kernel_size (list[int]): kernel size per stage for patch embedding.
        patch_stride (list[int]): stride per stage for patch embedding.
        patch_padding (list[int]): padding per stage for patch embedding.
        mlp_ratios (list[int]): MLP expansion ratios.
        use_hybrid_stem (bool): use convolutional stem before transformer.

    Input shape:
        (B, 1, D, H, W)  — e.g. (1, 1, 32, 160, 208)

    Output:
        Tuple of two tensors:
            - logits: (B, num_classes, D, H, W)
            - aux:    (B, num_classes, D, H, W)
    """
  
    def __init__(self, in_channels=1, num_classes=3, embed_dims=[32, 64, 160, 256], depths=[2,2,2,2],
                 num_heads=[1,2,5,8], sr_ratios=[4,2,1,1],
                 patch_kernel_size=[7,3,3,3], patch_stride=[4,2,2,2], patch_padding=[3,1,1,1],
                 mlp_ratios=[4,4,4,4], use_hybrid_stem=True, decoder_C=128, aspp_rates=(1,2,3), fuse_all=True):
        super().__init__()

        self.encoder = MixVisionTransformer(
            in_channels, embed_dims, num_heads, depths,
            sr_ratios, patch_kernel_size, patch_stride, patch_padding, mlp_ratios,use_hybrid_stem=use_hybrid_stem
        )

        # NEW DECODER (uses raw conv-feature maps from encoder)
        self.decoder = ASPPFPNDecoderHead(
            in_channels=embed_dims,            # [c1,c2,c3,c4]
            num_classes=num_classes,
            C=decoder_C,
            aspp_rates=aspp_rates,
            fuse_all=fuse_all
        )

    def forward(self, x):
        input_shape = x.shape[2:]  # (D, H, W)
        c1, c2, c3, c4 = self.encoder(x)
        logits, aux = self.decoder(c1, c2, c3, c4)
        logits = F.interpolate(logits, size=input_shape, mode="trilinear", align_corners=False)
        aux    = F.interpolate(aux,    size=input_shape, mode="trilinear", align_corners=False)
        return logits, aux
