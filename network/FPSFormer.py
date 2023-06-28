import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Conv2Module, ConvUp
from .init_weights import init_weights
from torchvision import models
from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from .DualAttn import *
from .RRM import *


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class FPSFormer(nn.Module):
    def __init__(self, n_channels=3, phi='b1', is_deconv=True,
                is_batchnorm=True, dropout_ratio=0.1):
        super(FPSFormer, self).__init__()      

        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone   = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained=False)
        self.embedding_dim   = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]

        embedding_dim=self.embedding_dim
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        self.n_channels = n_channels
      
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
      
        filters = [64, 128, 320, 512, 1024] 
        
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.center = Conv2Module(filters[3], filters[4], self.is_batchnorm)

        self.DA3=DualAttention(filters[2], filters[2])
        self.DA4=DualAttention(filters[3], filters[3])

        # upsampling
        self.up_concat4 = ConvUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = ConvUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = ConvUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = ConvUp(filters[1], filters[0], self.is_deconv)
      
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim*4,
            c2=embedding_dim,
            k=1,
        )
        
        out_ch=1
        self.linear_pred    = nn.Conv2d(embedding_dim, out_ch, kernel_size=1)
        self.outconv1    = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.dropout        = nn.Dropout2d(dropout_ratio)

        self.refunet = RefUnet(1,64)
        
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        # MiT encoder
        outs = self.backbone.forward(inputs)
       
        c1, c2, c3, c4 = outs
        maxpool4 = self.maxpool4(c4)
        
        center = self.center(maxpool4)  
   
        c3=self.DA3(c3)
        c4=self.DA4(c4)
        
        # decoder
        up4 = self.up_concat4(center, c4)  
        up3 = self.up_concat3(up4, c3) 
        up2 = self.up_concat2(up3, c2)  
        up1 = self.up_concat1(up2, c1) 

        n, _, h, w = up4.shape
        
        _c4 = self.linear_c4(up4).permute(0,2,1).reshape(n, -1, up4.shape[2], up4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(up3).permute(0,2,1).reshape(n, -1, up3.shape[2], up3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(up2).permute(0,2,1).reshape(n, -1, up2.shape[2], up2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(up1).permute(0,2,1).reshape(n, -1, up1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)
        Sfuse = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        Sout=self.refunet(x)
        Sout = F.interpolate(Sout, size=(H, W), mode='bilinear', align_corners=True)
        
        d1 = self.dropout(up1)
        d1 = self.outconv1(d1)  
        S1= F.interpolate(d1, size=(H, W), mode='bilinear', align_corners=True)

        d2 = self.dropout(_c2)
        d2 = self.linear_pred(d2)  
        S2= F.interpolate(d2, size=(H, W), mode='bilinear', align_corners=True)

        d3 = self.dropout(_c3)
        d3 = self.linear_pred(d3)  
        S3= F.interpolate(d3, size=(H, W), mode='bilinear', align_corners=True)

        d4 = self.dropout(_c4)
        d4 = self.linear_pred(d4)  
        S4= F.interpolate(d4, size=(H, W), mode='bilinear', align_corners=True)
        
        return F.sigmoid(Sout), F.sigmoid(Sfuse), F.sigmoid(S1), F.sigmoid(S2), F.sigmoid(S3), F.sigmoid(S4)  

