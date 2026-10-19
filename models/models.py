import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from .helpers import Site_Resblock, Brain_Resblock, UpResblock, SpatialTransformer2d

def get_network_class(network_name):
    """Return the algorithm class with the given name."""
    if network_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(network_name))
    return globals()[network_name]


class MultiscaleEncoder(nn.Module):
    def __init__(self, configs, hparams):
        super(MultiscaleEncoder,self).__init__()
        self.filter_size = hparams['filter_size']

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.S_downblock_1 = Site_Resblock(in_ch = configs.site_input_channels,out_ch = self.filter_size[0]//2,
                                           stride = 1,return_1strelu=False)
        self.S_downblock_2 = Site_Resblock(in_ch = self.filter_size[0]//2,out_ch = self.filter_size[0],
                                           stride = 1,return_1strelu=False)
        self.S_downblock_3 = Site_Resblock(in_ch = self.filter_size[0],out_ch = self.filter_size[0],
                                           stride = 1,return_1strelu=False)
        self.S_downblock_4 = Site_Resblock(in_ch = self.filter_size[0],out_ch = self.filter_size[1],
                                           stride = 1,return_1strelu=False)
        self.S_downblock_5 = Site_Resblock(in_ch = self.filter_size[1],out_ch = self.filter_size[1],
                                           stride = 1,return_1strelu=False)
        self.S_downblock_6 = Site_Resblock(in_ch = self.filter_size[1],out_ch = self.filter_size[1],
                                           stride = 1,return_1strelu=False)
        self.S_downblock_7 = Site_Resblock(in_ch = self.filter_size[1],out_ch = self.filter_size[1],
                                           stride = 1,return_1strelu=False)
        self.S_downblock_8 = Site_Resblock(in_ch = self.filter_size[1],out_ch = self.filter_size[1],
                                           stride = 1,return_1strelu=False)         

    def forward(self, slc):

        F1 = self.S_downblock_1(slc) 
        F2 = self.maxpool(F1) 

        F2 = self.S_downblock_2(F2)
        F3 = self.S_downblock_3(F2)
        F3 = self.maxpool(F3) 

        F3 = self.S_downblock_4(F3) 
        F4 = self.S_downblock_5(F3) 

        # Bottleneck
        F4 = self.S_downblock_6(F4)
        F4 = self.S_downblock_7(F4) 
        F4 = self.S_downblock_8(F4) 

        return {
            "F1": F1,
            "F2": F2,
            "F3": F3,
            "F4": F4
        }

class AuxiliaryDecoder(nn.Module):
    def __init__(self, configs, hparams):
        super(AuxiliaryDecoder,self).__init__()
        self.filter_size = hparams['filter_size'] 

        self.upsampling = nn.Upsample(scale_factor=2, mode="nearest")

        self.S_upblock_1 = Site_Resblock(in_ch = self.filter_size[1],out_ch = self.filter_size[1],
                                         stride = 1,return_1strelu=False)
        self.S_upblock_2 = Site_Resblock(in_ch = self.filter_size[1],out_ch = self.filter_size[1],
                                         stride = 1,return_1strelu=False)
        self.S_upblock_3 = Site_Resblock(in_ch = self.filter_size[1],out_ch = self.filter_size[1],
                                         stride = 1,return_1strelu=False)
        self.S_upblock_4 = Site_Resblock(in_ch = self.filter_size[1],out_ch = self.filter_size[1],
                                         stride = 1,return_1strelu=False)
        self.S_upblock_5 = Site_Resblock(in_ch = self.filter_size[1],out_ch = self.filter_size[0],
                                         stride = 1,return_1strelu=False)
        self.S_upblock_6 = Site_Resblock(in_ch = self.filter_size[0],out_ch = self.filter_size[0],
                                         stride = 1,return_1strelu=False)
        self.S_upblock_7 = Site_Resblock(in_ch = self.filter_size[0],out_ch = self.filter_size[0]//2,
                                         stride = 1,return_1strelu=False)
        self.S_upblock_8 = Site_Resblock(in_ch = self.filter_size[0]//2,out_ch = self.filter_size[0]//2,
                                         stride = 1,return_1strelu=False)

        self.outconv = nn.Conv2d(in_channels=self.filter_size[0]//2, out_channels=configs.site_output_channels,
                                 kernel_size=1, stride=1, padding=0, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, F4):

        x = self.S_upblock_1(F4) 
        x = self.S_upblock_2(x) 
        x = self.S_upblock_3(x) 

        x = self.S_upblock_4(x) 
        x = self.S_upblock_5(x) 

        x = self.upsampling(x) 

        x = self.S_upblock_6(x) 
        x = self.S_upblock_7(x) 

        x = self.upsampling(x) 

        x = self.S_upblock_8(x) 

        out = self.outconv(x)
        out = self.tanh(out)

        return out







class BrainEncoder(nn.Module):
    def __init__(self, configs, hparams):
        super(BrainEncoder,self).__init__()

        self.filter_size = hparams['filter_size']
        self.encoder_stride = hparams['stride_encoder']
        self.upsampling = nn.Upsample(scale_factor=2, mode="nearest")
        self.B_downblock_1 = Brain_Resblock(in_ch=configs.input_channels,out_ch=self.filter_size[0],
                                            stride=2) 
        self.B_downblock_2 = Brain_Resblock(in_ch=self.filter_size[0],out_ch=self.filter_size[1],
                                            stride=2) 
        self.B_downblock_3 = Brain_Resblock(in_ch=self.filter_size[1],out_ch=self.filter_size[2],
                                            stride=2) 
        self.B_downblock_4 = Brain_Resblock(in_ch=self.filter_size[2],out_ch=self.filter_size[3],
                                            stride=2) 
        self.B_bottleneck = Brain_Resblock(in_ch=self.filter_size[3],out_ch=self.filter_size[3],
                                            stride=1) 

        self.B_upblock_1 = Brain_Resblock(in_ch=self.filter_size[2]+self.filter_size[3],out_ch=self.filter_size[2],
                                            stride=1)  
        self.B_upblock_2 = Brain_Resblock(in_ch=self.filter_size[1]+self.filter_size[2],out_ch=self.filter_size[1],
                                            stride=1) 
        self.B_upblock_3 = Brain_Resblock(in_ch=self.filter_size[1]+self.filter_size[0],out_ch=self.filter_size[0],
                                            stride=1)
        self.instancenorm = nn.InstanceNorm2d(self.filter_size[0]) 
        
    def forward(self, slc):
        B1 = self.B_downblock_1(slc)
        B2 = self.B_downblock_2(B1) 
        B3 = self.B_downblock_3(B2) 
        B4 = self.B_downblock_4(B3) 
        B4 = self.B_bottleneck(B4) 

        B = self.upsampling(B4) 
        B = torch.cat([B, B3], dim=1)
        B = self.B_upblock_1(B)

        B = self.upsampling(B)
        B = torch.cat([B, B2], dim=1)
        B = self.B_upblock_2(B)

        B = self.upsampling(B)
        B = torch.cat([B, B1], dim=1)
        B = self.B_upblock_3(B)
        B = self.instancenorm(B)

        return B


class HarmonizationDecoder(nn.Module):

    def __init__(self, configs, hparams):
        super(HarmonizationDecoder, self).__init__()
        self.filter_size = hparams['filter_size']
        self.encoder_stride = hparams['stride_encoder']
        self.out_channel = configs.decoder_output_channels 

        self.B_downblock_2 = Brain_Resblock(in_ch=self.filter_size[0],out_ch=self.filter_size[1],
                                            stride=self.encoder_stride)
        self.B_downblock_3 = Brain_Resblock(in_ch=self.filter_size[1],out_ch=self.filter_size[2],
                                            stride=self.encoder_stride)
        self.B_downblock_4 = Brain_Resblock(in_ch=self.filter_size[2],out_ch=self.filter_size[3],
                                            stride=self.encoder_stride)
        self.B_downblock_2_res = Brain_Resblock(in_ch=self.filter_size[1], out_ch=self.filter_size[1], stride=1)
        self.B_downblock_3_res = Brain_Resblock(in_ch=self.filter_size[2], out_ch=self.filter_size[2], stride=1)
        self.B_downblock_4_res = Brain_Resblock(in_ch=self.filter_size[3], out_ch=self.filter_size[3], stride=1)

        self.S_context = nn.Sequential(
            Site_Resblock(in_ch=self.filter_size[1], out_ch=self.filter_size[1], stride=2),
            Site_Resblock(in_ch=self.filter_size[1], out_ch=self.filter_size[1], stride=1),
        )
        self.S3_context = nn.Sequential(
            Site_Resblock(in_ch=self.filter_size[1], out_ch=self.filter_size[1], stride=2),
            Site_Resblock(in_ch=self.filter_size[1], out_ch=self.filter_size[1], stride=1),
        )
        self.S2_context = nn.Sequential(
            Site_Resblock(in_ch=self.filter_size[0], out_ch=self.filter_size[0], stride=4),
            Site_Resblock(in_ch=self.filter_size[0], out_ch=self.filter_size[0], stride=1),
        )
        self.xattn_1 = SpatialTransformer2d(
            in_channels=self.filter_size[2], 
            context_dim=self.filter_size[1],
            num_heads=4,
            head_dim=32,
            num_layers=hparams.get('num_transformer_layer',1),
            dropout=0.0,
        )
        self.xattn_2 = SpatialTransformer2d(
            in_channels=self.filter_size[1], 
            context_dim=self.filter_size[1],
            num_heads=4,
            head_dim=16,
            num_layers=hparams.get('num_transformer_layer',1),
            dropout=0.0,
        )
        self.xattn_3 = SpatialTransformer2d(
            in_channels=self.filter_size[0], 
            context_dim=self.filter_size[0],
            num_heads=4,
            head_dim=8,
            num_layers=hparams.get('num_transformer_layer',1),
            dropout=0.0,
        )        
        self.upsampling = nn.Upsample(scale_factor=2, mode="nearest")

        self.B_upblock_1a = UpResblock(in_ch=self.filter_size[3] + self.filter_size[2], 
                                        out_ch=self.filter_size[2], stride=1) 

        self.B_upblock_1b = UpResblock(in_ch=self.filter_size[2], out_ch=self.filter_size[2], stride=1)

        self.B_upblock_2a = UpResblock(in_ch=self.filter_size[2] + self.filter_size[1], 
                                        out_ch=self.filter_size[1], stride=1) 

        self.B_upblock_2b = UpResblock(in_ch=self.filter_size[1], out_ch=self.filter_size[1], stride=1)

        self.B_upblock_3a = UpResblock(in_ch=self.filter_size[1] + self.filter_size[0], 
                                        out_ch=self.filter_size[0], stride=1) 

        self.B_upblock_3b = UpResblock(in_ch=self.filter_size[0], out_ch=self.filter_size[0], stride=1)

        self.final_block = UpResblock(in_ch=self.filter_size[0], out_ch=self.filter_size[0], stride=1)
        self.final_conv = nn.Conv2d(
            in_channels=self.filter_size[0], out_channels=self.out_channel,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.tanh = nn.Tanh()

    def B_downblocks(self, B_out1):
        B_out2 = self.B_downblock_2_res(self.B_downblock_2(B_out1))
        B_out3 = self.B_downblock_3_res(self.B_downblock_3(B_out2))
        B_out4 = self.B_downblock_4_res(self.B_downblock_4(B_out3))
        return B_out2, B_out3, B_out4

    def _to_context(self, feat):
        b, c, h, w = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous()

    def upsample_path(self, reference_features, B_out1, B_out2, B_out3, B_out4):

        ctx_f4 = self._to_context(self.S_context(reference_features["F4"])) 
        ctx_f3 = self._to_context(self.S3_context(reference_features["F3"]))
        ctx_f2 = self._to_context(self.S2_context(reference_features["F2"]))
        # Up 1
        x = self.upsampling(B_out4)
        x = torch.cat([x, B_out3], dim=1)
        x = self.B_upblock_1a(x)
        x = self.xattn_1(x, ctx_f4)
        x = self.B_upblock_1b(x)

        # Up 2
        x = self.upsampling(x)
        x = torch.cat([x, B_out2], dim=1)
        x = self.B_upblock_2a(x)
        x = self.xattn_2(x, ctx_f3) 
        x = self.B_upblock_2b(x)

        # Up 3
        x = self.upsampling(x)
        x = torch.cat([x, B_out1], dim=1)
        x = self.B_upblock_3a(x)
        x = self.xattn_3(x, ctx_f2)
        x = self.B_upblock_3b(x)

        # Final
        x = self.upsampling(x)
        x = self.final_block(x)
        x = self.final_conv(x)
        x = self.tanh(x)

        return x

    def forward(self, reference_features, B_out1):
        B_out2, B_out3, B_out4 = self.B_downblocks(B_out1)
        return self.upsample_path(reference_features, B_out1, B_out2, B_out3, B_out4)