import torch
from torch import nn
import torch.nn.functional as F
import math
from monai.networks.blocks import CrossAttentionBlock, MLPBlock

class MONAITransformerBlock(nn.Module):

    def __init__(
        self,
        inner_dim: int,
        context_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
    ):
        super(MONAITransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(inner_dim)
        self.attn = CrossAttentionBlock(
            hidden_size=inner_dim,
            num_heads=num_heads,
            hidden_input_size=inner_dim,
            context_input_size=context_dim,
            dim_head=head_dim,
            dropout_rate=dropout,
            use_flash_attention=True,
        )

        self.norm2 = nn.LayerNorm(inner_dim)

        self.ff = MLPBlock(
            hidden_size=inner_dim,
            mlp_dim=inner_dim * 4,
            act="GEGLU",
            dropout_rate=dropout,
        )

    def forward(self, x, context):
        

        x = self.attn(self.norm1(x), context=context) + x
        x = self.ff(self.norm2(x)) + x
        return x


class SpatialTransformer2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        context_dim: int,
        num_heads: int,
        head_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super(SpatialTransformer2d, self).__init__()
        inner_dim = num_heads * head_dim

        self.norm = nn.LayerNorm(in_channels)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, bias=True)

        self.blocks = nn.ModuleList(
            [
                MONAITransformerBlock(
                    inner_dim=inner_dim,
                    context_dim=context_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, bias=True)

    def forward(self, x, context):
        b, c, h, w = x.shape
        residual = x

        x = x.permute(0, 2, 3, 1).contiguous() 
        x = self.norm(x)                        
        x = x.permute(0, 3, 1, 2).contiguous()  

        x = self.proj_in(x) 
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, -1).contiguous()

        for block in self.blocks:
            x = block(x, context)

        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous() 
        x = self.proj_out(x)
        return x + residual


class Conv2dSame(torch.nn.Conv2d):
    """
    From https://github.com/pytorch/pytorch/issues/67551
    """
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Site_Resblock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, negative_slope=0.2,return_1strelu=False):
        super(Site_Resblock,self).__init__()
        self.conv1 = Conv2dSame(in_ch, out_ch, kernel_size = 4, stride = stride, bias=True) 
        self.conv2 =  Conv2dSame(out_ch, out_ch, kernel_size=4, stride=1, bias=True)
        self.leakyrelu = nn.LeakyReLU(negative_slope = negative_slope, inplace=True)
        self.return_1strelu = return_1strelu
        self.use_identity_skip = (stride == 1 and in_ch == out_ch)
        if not self.use_identity_skip:
            self.skipconv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=True)
    def forward(self, x_in):
        if self.use_identity_skip:
            skip_x = x_in
        else:
            skip_x = self.skipconv(x_in)

        conv_x = self.conv1(x_in)

        if self.return_1strelu:
            conv_x_ = self.leakyrelu(conv_x)
            conv_x = self.conv2(conv_x_)
        else:
            conv_x = self.leakyrelu(conv_x)
            conv_x = self.conv2(conv_x)
        
        out = conv_x + skip_x
        out = self.leakyrelu(out)
        if self.return_1strelu:
            return out,conv_x_
        else:    
            return out


class Brain_Resblock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, negative_slope=0.2,num_groups=8):
        super(Brain_Resblock,self).__init__()
        self.conv1 = Conv2dSame(in_ch, out_ch, kernel_size = 4, stride = stride, bias=True)
        self.conv2 = Conv2dSame(out_ch, out_ch, kernel_size=4, stride=1, bias=True)
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_ch)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_ch)
        self.leakyrelu = nn.LeakyReLU(negative_slope = negative_slope, inplace=True)
        self.use_identity_skip = (stride == 1 and in_ch == out_ch)
        if not self.use_identity_skip:
            self.skipconv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=True)

    def forward(self, x_in):


        skip_x = x_in if self.use_identity_skip else self.skipconv(x_in)

        conv_x = self.conv1(x_in)
        conv_x = self.norm1(conv_x)
        conv_x = self.leakyrelu(conv_x)
        conv_x = self.conv2(conv_x)
        conv_x = self.norm2(conv_x)
        out = conv_x + skip_x

        out = self.leakyrelu(out)
        
        return out


def _apply_layernorm_2d(norm, x):
    x = x.permute(0, 2, 3, 1).contiguous()
    x = norm(x)
    return x.permute(0, 3, 1, 2).contiguous()

class UpResblock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, negative_slope=0.2):
        super(UpResblock, self).__init__()
        self.conv1 = Conv2dSame(in_ch, out_ch, kernel_size=4, stride=stride, bias=True)
        self.conv2 = Conv2dSame(out_ch, out_ch, kernel_size=4, stride=1, bias=True)
        self.leakyrelu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.norm1 = nn.LayerNorm(out_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        self.use_identity_skip = (stride == 1 and in_ch == out_ch)
        if not self.use_identity_skip:
            self.skipconv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=True)

    def forward(self, x_in):
        skip_x = x_in if self.use_identity_skip else self.skipconv(x_in)
        conv_x = _apply_layernorm_2d(self.norm1, self.conv1(x_in))
        conv_x = self.leakyrelu(conv_x)
        conv_x = _apply_layernorm_2d(self.norm2, self.conv2(conv_x))
        out = self.leakyrelu(conv_x + skip_x)
        return out



class PointwiseMLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=1024):
        super(PointwiseMLP,self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True) 
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.act = nn.ReLU()


    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x