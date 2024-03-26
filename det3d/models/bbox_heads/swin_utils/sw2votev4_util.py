import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import pdb


class MLP(nn.Module):
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


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.register_parameter("tau", nn.Parameter(torch.ones(1, num_heads, 1, 1)))
        self.rpe = nn.Sequential(nn.Conv2d(2, 16, kernal_size=1, stride=1, bias=True), 
                                 nn.ReLU(), 
                                 nn.Conv2d(16, num_heads, kernal_size=1, stride=1, bias=True))
        self.vote_mlp = nn.Sequential(nn.Conv2d(3, 16, kernal_size=1, stride=1, bias=True), 
                                 nn.ReLU(), 
                                 nn.Conv2d(16, dim, kernal_size=1, stride=1, bias=True))
        
    def forward(self, x, mask=None, pos_embed=None, vote_embed=None):
        B_, N, C = x.shape
        _, _, C_embed = pos_embed.shape

        vote_embed = vote_embed.permute(0, 2, 1).contiuous()
        vote_embed = self.vote_mlp(vote_embed)
        vote_embed = vote_embed.reshape(vote_embed.shape[0], self.num_heads, self.dim // self.num_heads, -1)
        vote_embed = vote_embed.permute(0, 1, 3, 2).contiguous()

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q += vote_embed
        k += vote_embed
        v += vote_embed

        attn = torch.einsum('bhnd,bhmd->bhnm', q, k) * self.scale // torch.maixmum(torch.norm(q, dim=-1, keepdim=True)*torch.norm(k, dim=-1, keepdim=True).transpose(-2, -1), torch.tensor(1e-6))
        attn = attn / self.tau.clamp(min=0.01)

        pos_embed = pos_embed.permute(0, 2, 1).contiguous()
        rpe = pos_embed[:,:,:,None] - pos_embed[:,:,None,:]
        rpe = self.rpe(rpe)
        attn = attn + rpe

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        
        attn = self.attn_drop(attn)
        
        x = (attn @ (v + vote_embed)).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads = num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix, polar_embed=None, vote_embed=None):
        B, L, C = x.shape
        _, _, C_embed = polar_embed.shape
        _, _, C_vote = vote_embed.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        polar_embed = polar_embed.view(B, H, W, C_embed)
        vote_embed = vote_embed.view(B, H, W, C_vote)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, [0, 0, pad_l, pad_r, pad_t, pad_b])

        _, Hp, Wp, _ = x.shape

        polar_embed = F.pad(polar_embed, [0, 0, pad_l, pad_r, pad_t, pad_b])
        vote_embed = F.pad(vote_embed, [0, 0, pad_l, pad_r, pad_t, pad_b])

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix

            shifted_polar_embed = torch.roll(polar_embed, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_vote_embed = torch.roll(vote_embed, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

            shifted_polar_embed = polar_embed
            shifted_vote_embed = vote_embed

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        polar_embed_windows = window_partition(shifted_polar_embed, self.window_size)
        polar_embed_windows = polar_embed_windows.view(-1, self.window_size * self.window_size, C_embed)

        vote_embed_windows = window_partition(shifted_vote_embed, self.window_size)
        vote_embed_windows = vote_embed_windows.view(-1, self.window_size * self.window_size, C_vote)

        attn_windows = self.attn(x_windows, mask=attn_mask, pos_embed=polar_embed_windows, vote_embed=vote_embed_windows)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    def __init__(self, 
                 dim, 
                 depth, 
                 num_heads, 
                 window_size=7, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop=0., 
                 attn_drop=0., 
                 drop_path=0., 
                 norm_layer=nn.LayerNorm, 
                 downsample=None, 
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, 
                                 num_heads=num_heads, 
                                 window_size=window_size, 
                                 shift_size=0 if (i%2==0) else window_size // 2, 
                                 mlp_ratio=mlp_ratio, 
                                 qkv_bias=qkv_bias, 
                                 qk_scale=qk_scale, 
                                 drop=drop, 
                                 attn_drop=attn_drop, 
                                 drop_path=drop_path[i] if isinstance(drop_path,list) else drop_path, 
                                 norm_layer=norm_layer)
            for i in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W, polar_embed=None, vote_embed=None):
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device, dtype=torch.bool)
        h_slices = (slice(0, -self.window_size), 
                    slice(-self.window_size, -self.shift_size), 
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), 
                    slice(-self.window_size, -self.shift_size), 
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask, polar_embed, vote_embed)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W
        
class SwinTransformer(nn.Module):
    def __init__(self, 
                 embed_dim=256, 
                 depth=[4], 
                 num_heads=[4], 
                 window_size=7, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(0),
                 use_checkpoint=False,
                 in_ch=256,
                 use_patch_embed=False):
        super().__init__()

        self.num_layers = len(depth)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices

        self.in_ch = in_ch
        self.use_patch_embed = use_patch_embed
        self.patch_embed = None
        if self.use_patch_embed:
            self.patch_embed = PatchEmbed(patch_size=1, 
                                          in_chans=self.in_ch, 
                                          embed_dim=self.embed_dim, 
                                          norm_layer=norm_layer if self.patch_norm else None
                                          )
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               depth=depth[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias,
                               qk_scale=qk_scale,
                               drop=drop_rate,
                               attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depth[:i_layer]):sum(depth[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            
            num_features = [int(embed_dim * 2 ** i_layer) for i in range(self.num_layers)]
            self.num_features = num_features

            i_layer = 0
            layer = norm_layer(num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    def forward(self, x, polar_embed=None, vote_embed=None):
        if self.use_patch_embed:
            x = self.patch_embed(x)

        Wh, Ww = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        polar_embed = polar_embed.flatten(2).transpose(1, 2)
        vote_embed = vote_embed.flatten(2).transpose(1, 2)

        outs = []

        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww, polar_embed, vote_embed)

            i = 0
            norm_layer = getattr(self, f"norm{i}")
            x_out = norm_layer(x_out)

            out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
            outs.append(out)

        return outs
    
class PatchEmbed(nn.Module):
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
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, [0, self.patch_size[1] - W % self.patch_size[1]])
        if H % self.patch_size[0] != 0:
            x = F.pad(x, [0, 0, 0, self.patch_size[0] - H % self.patch_size[0]])

        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).reshape(-1, self.embed_dim, Wh, Ww)

        return x
