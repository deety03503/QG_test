import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath
import timm
from functools import partial
from collections import OrderedDict
from timm.models.vision_transformer import PatchEmbed


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x)
        k = self._shape(self.k_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(self.v_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim).transpose(1, 2).reshape(B, N, C)
        x = self.proj(attn_output)
        x = self.proj_drop(x)
        return x


class Adapter(nn.Module):
    def __init__(self, config=None, d_model=768, bottleneck=64, dropout=0.0, adapter_scalar="1.0"):
        super().__init__()
        self.n_embd = config.d_model if (config and hasattr(config, 'd_model')) else d_model
        self.down_size = config.ffn_num if (config and hasattr(config, 'ffn_num')) else bottleneck

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.dropout = dropout
        self.scale = float(adapter_scalar) if isinstance(adapter_scalar, (str, float, int)) else 1.0

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        return up * self.scale


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None):
        super().__init__()
        self.config = config
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.layer_id = layer_id
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

        # ModuleList chứa các adapter cho từng task
        self.adapters = nn.ModuleList([
            Adapter(
                self.config,
                d_model=dim,
                bottleneck=getattr(config, 'ffn_num', 64),
                dropout=getattr(config, 'ffn_adapter_dropout', 0.0),
                adapter_scalar=getattr(config, 'ffn_adapter_scalar', '1.0'),
            )
        ])

    def add_adapter(self):
        # Freeze các adapter cũ
        for adapter in self.adapters:
            for p in adapter.parameters():
                p.requires_grad = False
            adapter.eval()

        # Tạo adapter mới
        new_adapter = Adapter(
            self.config,
            d_model=self.fc1.in_features,
            bottleneck=getattr(self.config, 'ffn_num', 64),
            dropout=getattr(self.config, 'ffn_adapter_dropout', 0.0),
            adapter_scalar=getattr(self.config, 'ffn_adapter_scalar', '1.0'),
        )
        self.adapters.append(new_adapter)

    def forward(self, x, task_idx=None, adapter_weights=None):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        residual = x
        mlp_input = self.norm2(x)
        mlp_out = self.mlp_drop(self.fc2(self.act(self.fc1(mlp_input))))

        if task_idx is not None:
            # Chọn adapter cụ thể (ví dụ task_idx = 0 cho Base Adapter A1)
            adapt_out = self.adapters[task_idx](mlp_input)
        elif adapter_weights is not None:
            # Trọng số kết hợp các adapters (Task-interaction fusion)
            adapt_out = 0
            for i, w in enumerate(adapter_weights):
                if i < len(self.adapters):
                    adapter_out = self.adapters[i](mlp_input)
                    if torch.is_tensor(w) and w.ndim == 1:
                        w = w.view(-1, 1, 1)
                    adapt_out = adapt_out + w * adapter_out
        else:
            # Mặc định dùng adapter cuối cùng (task hiện tại)
            adapt_out = self.adapters[-1](mlp_input)

        x = residual + self.drop_path(mlp_out + adapt_out)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', tuning_config=None):
        super().__init__()
        self.tuning_config = tuning_config
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                  config=tuning_config, layer_id=i)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.global_pool = global_pool

    def add_adapter(self):
        for blk in self.blocks:
            blk.add_adapter()

    def forward_features(self, x, task_idx=None, adapter_weights=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, task_idx=task_idx, adapter_weights=adapter_weights)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x, task_idx=None, adapter_weights=None):
        return self.forward_features(x, task_idx=task_idx, adapter_weights=adapter_weights)


def vit_base_patch16_224_adapter(pretrained=True, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    checkpoint_model = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)
    state_dict = checkpoint_model.state_dict()

    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = qkv_weight[:768]
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = qkv_weight[768:768 * 2]
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = qkv_weight[768 * 2:]
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = qkv_bias[:768]
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = qkv_bias[768:768 * 2]
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = qkv_bias[768 * 2:]

    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)

    # Freeze backbone, trainable adapter only
    for name, p in model.named_parameters():
        if 'adapters' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model


def vit_base_patch16_224_in21k_adapter(pretrained=True, **kwargs):
    return vit_base_patch16_224_adapter(pretrained=pretrained, **kwargs)



