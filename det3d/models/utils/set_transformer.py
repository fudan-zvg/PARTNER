import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def img2windows(img, H_sp, W_sp):
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H, W // window_size, window_size, C)
    windows = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, H, window_size, C)
    return windows


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SetAttention(nn.Module):
    def __init__(self, dim, resolution, H_sp=144, W_sp=1, H=4, W=8, dim_out=None, num_heads=8, drop=0., attn_drop=0., drop_path=0.,
                 qkv_bias=True, qk_scale=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_ratio=4., shift=True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.H = H
        self.W = W
        self.shift_size = self.W // 2 if shift else 0

        self.norm1 = norm_layer(dim)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

        self.pos_embedding_cart = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, num_heads, kernel_size=1))

        self.range_attn = RangeAttention(dim, resolution=(self.H, resolution[1]),
                H_sp=self.H, W_sp=self.W, num_heads=num_heads, dim_out=dim,
                qk_scale=qk_scale, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                act_layer=act_layer, norm_layer=norm_layer, mlp_ratio=mlp_ratio)

        self.sector_attn1 = SectorAttention(dim, resolution=resolution,
                                         H_sp=self.H_sp, W_sp=self.W_sp, H=self.H, num_heads=num_heads, dim_out=dim,
                                         qk_scale=qk_scale, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         act_layer=act_layer, norm_layer=norm_layer, mlp_ratio=mlp_ratio)

        self.sector_attn2 = SectorAttentionV2(dim, resolution=resolution,
                                           H_sp=self.H_sp, W_sp=self.W_sp, H=self.H, num_heads=num_heads, dim_out=dim,
                                           qk_scale=qk_scale, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                                           drop_path=drop_path,
                                           act_layer=act_layer, norm_layer=norm_layer, mlp_ratio=mlp_ratio)

    def im2cswin(self, x):
        B, N, C = x.shape
        H, W = self.resolution
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, x, x_pos):
        """
        x: B, L, C
        """

        shortcut = x
        x = self.norm1(x)

        B, L, C = x.shape
        H, W = self.resolution
        x = x.view(B, H, W, C).contiguous()

        if self.shift_size > 0:
            x = torch.roll(x, shifts=-self.shift_size, dims=2)
            x_pos = torch.roll(x_pos, shifts=-self.shift_size, dims=2)

        # keypoint initialization
        s = x.mean(dim=3)
        s = s.permute(0, 2, 1)
        local_max = torch.zeros_like(s).to(s.device)
        kernel_size = 3
        padding = kernel_size // 2
        local_max_inner = F.max_pool1d(s, kernel_size=kernel_size, stride=1, padding=0)
        local_max[:, :, padding:(-padding)] = local_max_inner
        s = s * (local_max == s)
        s = s.permute(0, 2, 1)
        top_idx = s.argsort(dim=1, descending=True)[:, :self.H, :]
        s = x.gather(index=top_idx[:, :, :, None].expand(-1, -1, -1, C), dim=1)
        s = s.view(B, -1, C)
        s_pos = x_pos.gather(index=top_idx[:, :, :, None].expand(-1, -1, -1, x_pos.shape[3]), dim=1)

        x = x.view(B, L, C).contiguous()

        s = self.sector_attn1(s, x, s_pos, x_pos)

        s = self.range_attn(s, s_pos)

        x = self.sector_attn2(s, x, s_pos, x_pos)

        if self.shift_size > 0:
            x = x.view(B, H, W, C).contiguous()
            x = torch.roll(x, shifts=(self.shift_size), dims=2)
            x = x.view(B, -1, C).contiguous()

        x = self.proj(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class RangeAttention(nn.Module):
    def __init__(self, dim, resolution, H_sp=4, W_sp=8, dim_out=None, num_heads=8, drop=0., attn_drop=0., drop_path=0.,
                 qkv_bias=True, qk_scale=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.H_sp = H_sp
        self.W_sp = W_sp

        self.norm1 = norm_layer(dim)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

        self.pos_embedding_cart = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, num_heads, kernel_size=1))


    def im2cswin(self, x):
        B, N, C = x.shape
        H, W = self.resolution
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, x, pos):
        """
        x: B, L, C
        """
        # q, k, v = qkv[0], qkv[1], qkv[2]
        H, W = self.resolution
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)

        pos = pos.view(B, H // self.H_sp, self.H_sp, W // self.W_sp, self.W_sp, 2)
        pos = pos.permute(0, 1, 3, 5, 2, 4).contiguous().reshape(-1, 2, self.H_sp * self.W_sp)
        relative_pos = pos[:, :, :, None] - pos[:, :, None, :]

        relative_pos = relative_pos.view(relative_pos.shape[0], 2, -1)
        relative_pos = self.pos_embedding_cart(relative_pos)
        relative_pos = relative_pos.view(-1, self.num_heads, self.H_sp * self.W_sp, self.H_sp * self.W_sp)

        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v = self.im2cswin(v)

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn += relative_pos

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v)  # + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        x = self.proj(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SectorAttention(nn.Module):
    def __init__(self, dim, resolution, H_sp=4, W_sp=8, H=4, dim_out=None, num_heads=8, drop=0., attn_drop=0., drop_path=0.,
                 qkv_bias=True, qk_scale=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.H = H

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

        self.pos_embedding_cart = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, num_heads, kernel_size=1))

    def im2cswin(self, x):
        B, N, C = x.shape
        H, W = self.resolution
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, s, x, s_pos, x_pos):
        """
        x: B, L, C
        """

        H, W = self.resolution
        B, L, C = x.shape

        x_pos = x_pos.view(B, H // self.H_sp, self.H_sp, W // self.W_sp, self.W_sp, 2)
        x_pos = x_pos.permute(0, 1, 3, 5, 2, 4).contiguous().reshape(-1, 2, self.H_sp * self.W_sp)

        s_pos = s_pos.permute(0, 2, 3, 1).contiguous().reshape(-1, 2, self.H * self.W_sp)

        relative_pos = s_pos[:, :, :, None] - x_pos[:, :, None, :]

        relative_pos = relative_pos.view(relative_pos.shape[0], 2, -1)
        relative_pos = self.pos_embedding_cart(relative_pos)
        relative_pos = relative_pos.view(-1, self.num_heads, self.H, self.H_sp)

        shortcut = s
        q = self.proj_q(s)
        k = self.proj_k(x)
        v = self.proj_v(x)

        q = q.view(B, C, self.H, W)
        q = q.view(B, self.num_heads, C // self.num_heads, self.H, W // self.W_sp, self.W_sp)
        q = q.permute(0, 4, 1, 3, 5, 2).contiguous()
        q = q.reshape(-1, self.num_heads, self.H * self.W_sp, C // self.num_heads)
        k = self.im2cswin(k)
        v = self.im2cswin(v)

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn += relative_pos

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        s = (attn @ v)  # + lepe
        s = s.transpose(1, 2).reshape(-1, self.H * self.W_sp, C)  # B head N N @ B head N C
        s = windows2img(s, self.H, self.W_sp, self.H, W).view(B, -1, C)  # B H' W' C

        s = self.proj(s)
        s = shortcut + self.drop_path(s)
        s = s + self.drop_path(self.mlp(self.norm2(s)))

        return s


class SectorAttentionV2(nn.Module):
    def __init__(self, dim, resolution, H_sp=4, W_sp=8, H=4, dim_out=None, num_heads=8, drop=0., attn_drop=0., drop_path=0.,
                 qkv_bias=True, qk_scale=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.H = H

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.pos_embedding_cart = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, num_heads, kernel_size=1))

    def im2cswin(self, x):
        B, N, C = x.shape
        H, W = self.resolution
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, s, x, s_pos, x_pos):
        """
        x: B, L, C
        """

        H, W = self.resolution
        B, L, C = x.shape

        x_pos = x_pos.view(B, H // self.H_sp, self.H_sp, W // self.W_sp, self.W_sp, 2)
        x_pos = x_pos.permute(0, 1, 3, 5, 2, 4).contiguous().reshape(-1, 2, self.H_sp * self.W_sp)

        s_pos = s_pos.permute(0, 2, 3, 1).contiguous().reshape(-1, 2, self.H * self.W_sp)

        relative_pos = x_pos[:, :, :, None] - s_pos[:, :, None, :]

        relative_pos = relative_pos.view(relative_pos.shape[0], 2, -1)
        relative_pos = self.pos_embedding_cart(relative_pos)
        relative_pos = relative_pos.view(-1, self.num_heads, self.H_sp, self.H)

        q = self.proj_q(x)
        k = self.proj_k(s)
        v = self.proj_v(s)

        q = self.im2cswin(q)

        k = k.view(B, C, self.H, W)
        k = k.view(B, self.num_heads, C // self.num_heads, self.H, W // self.W_sp, self.W_sp)
        k = k.permute(0, 4, 1, 3, 5, 2).contiguous()
        k = k.reshape(-1, self.num_heads, self.H * self.W_sp, C // self.num_heads)

        v = v.view(B, C, self.H, W)
        v = v.view(B, self.num_heads, C // self.num_heads, self.H, W // self.W_sp, self.W_sp)
        v = v.permute(0, 4, 1, 3, 5, 2).contiguous()
        v = v.reshape(-1, self.num_heads, self.H * self.W_sp, C // self.num_heads)

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn += relative_pos

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v)  # + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C


        return x


class SetBlock(nn.Module):
    def __init__(self, in_dim, embed_dim_scale, reso, num_heads,
                 H_sp=4, W_sp=4, H=4, W=8, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 pos=None, shift=True):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim_scale = embed_dim_scale
        self.embed_dim = self.in_dim // self.embed_dim_scale
        self.num_heads = num_heads
        self.patches_resolution = reso

        self.pos_cart = pos[..., :2]

        self.patch_norm = True

        if self.embed_dim_scale == 1:
            self.in_patch_embed = None
            self.out_patch_embed = None
        else:
            self.in_patch_embed = PatchEmbed(patch_size=1, in_chans=self.in_dim, embed_dim=self.embed_dim, norm_layer=norm_layer if self.patch_norm else None)
            self.out_patch_embed = PatchEmbed(patch_size=1, in_chans=self.embed_dim, embed_dim=self.in_dim, norm_layer=norm_layer if self.patch_norm else None)

        self.attns = SetAttention(
                self.embed_dim, resolution=self.patches_resolution,
                H_sp=H_sp, W_sp=W_sp, H=H, W=W, num_heads=num_heads, dim_out=self.embed_dim,
                qk_scale=qk_scale, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                act_layer=act_layer, norm_layer=norm_layer, mlp_ratio=mlp_ratio, shift=shift)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        if self.in_patch_embed:
            x = self.in_patch_embed(x.permute(0, 2, 1).contiguous().reshape(B, self.in_dim, H, W)).flatten(2).permute(0, 2, 1).contiguous()

        pos_cart = self.pos_cart.to(x.device)
        pos_cart = pos_cart.repeat(B, 1, 1, 1)

        x = self.attns(x, pos_cart)

        if self.out_patch_embed:
            x = self.out_patch_embed(x.permute(0, 2, 1).contiguous().reshape(B, self.embed_dim, H, W)).flatten(2).permute(0, 2, 1).contiguous()

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x
