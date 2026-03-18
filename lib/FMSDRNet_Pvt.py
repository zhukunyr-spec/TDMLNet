import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import *
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_unknown_tensor_from_pred
from einops import rearrange
import numbers
from lib.ForeCon import ForeCon


class MOLM(nn.Module):
    
    def __init__(self, x, y):
        super(MOLM, self).__init__()
        self.moconv = MOConv(in_channels=x, out_channels=y, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False)
        self.Conv = nn.Conv2d(x, y, kernel_size=3, stride=1, padding=1)
        self.atrConv = nn.Sequential(
            nn.Conv2d(x, y, kernel_size=3, dilation=2, padding=2, stride=1), nn.BatchNorm2d(y), nn.PReLU()
        )
        self.conv = nn.Conv2d(32,64,1)
        self.conv2d = nn.Conv2d(y*3, y, kernel_size=3, stride=1, padding=1)
        self.bn2d = nn.BatchNorm2d(y)


    def forward(self, f):
        p1 = self.Conv(f)
        p2 = self.moconv(f)
        p3 = self.atrConv(f)
        p  = torch.cat((p1, p2, p3), 1)
        p  = F.relu(self.bn2d(self.conv2d(p)), inplace=True)

        return p + self.conv(f)



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

class SAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(SAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(nn.ReLU(inplace=True))
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.SA = SpatialAttention(7)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        att = self.SA(res)
        res = res * att
        return res + x



class FFGM(nn.Module):
    def __init__(self, n_feat):
        super(FFGM, self).__init__()
        self.forecon = ForeCon(dim=n_feat)
        self.local = Local(dim=n_feat)
        #self.pos = Positioning(channel=n_feat)
        self.conv = nn.Conv2d(n_feat*2, n_feat, kernel_size=1)
    def forward(self, x):
        ## 得到前景信息
        x_forecon = self.forecon(x)
        x_forecon_back = 1-x_forecon ## 背景信息
        x_local  = self.local(x)
        x = torch.cat([x_forecon, x_forecon_back], dim=1)
        x = self.conv(x)
        x_ = x + x_local
        return x_

class Adapter(nn.Module):
    def __init__(self,dim=128):
        super(Adapter, self).__init__()

        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )


    def forward(self, x):
        B,C,H,W = x.shape
        x = x.reshape(B,C,-1).permute(0,2,1)
        prompt = self.prompt_learn(x)
        promped = x + prompt
        x = promped.permute(0,2,1).reshape(B,C,H,W)
        return x
    
class AFRM(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(AFRM, self).__init__()

        self.cab = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.sab = SAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        

        self.confidence = nn.Parameter(torch.full((n_feat, 1, 1), 0.5))
        self.out = nn.Sequential(
            nn.Conv2d(n_feat * 3, n_feat, 1),
            nn.SiLU()
        )
        self.out2 = nn.Sequential(
            nn.Conv2d(n_feat * 2, n_feat, 1),
            nn.SiLU()
        )
    def forward(self, x):
        x1 = self.cab(x)
        x2 = self.sab(x)
        
        conf = self.confidence.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        x_c = x * conf
        
        x_c = self.sab(x_c)
        
        out = torch.cat((x1 + x2, x_c, x_u), dim=1)
        
        out = self.out(out)
        
        out = torch.cat((x, out), dim=1)
        out = self.out2(out)
        return out + x


class FMSDRNet(nn.Module):
    def __init__(self, n_feat=64,kernel_size=3,reduction=4,bias=False,act=nn.PReLU(), train_mode=True):
        super(FMSDRNet, self).__init__()

        self.backbone = pvt_v2_b4()  # [64, 128, 320, 512]
        path = './lib/pvt_v2_b4.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer1_1 = BasicConv2d(64, n_feat//2, 1)
        self.Translayer2_1 = BasicConv2d(128, n_feat//2, 1)
        self.Translayer3_1 = BasicConv2d(320, n_feat//2, 1)
        self.Translayer4_1 = BasicConv2d(512, n_feat//2, 1)
        
        self.fe = MOLM(n_feat//2, n_feat)
 

        
        self.compress = nn.Conv2d(n_feat*3, n_feat*2, 1)
        self.expand = nn.Conv2d(n_feat, n_feat*2, 1)



        self.decoder4 = AFRM(n_feat * 2, kernel_size, reduction, bias=bias, act=act)
        self.decoder3 = AFRM(n_feat * 2, kernel_size, reduction, bias=bias, act=act)
        self.decoder2 = AFRM(n_feat * 2, kernel_size, reduction, bias=bias, act=act)
        self.decoder1 = AFRM(n_feat * 2, kernel_size, reduction, bias=bias, act=act)
        
        self.fl4 = FFGM(n_feat*2)
        self.fl3 = FFGM(n_feat*2)
        self.fl2 = FFGM(n_feat*2)
        self.fl1 = FFGM(n_feat*2)
        
        self.out_p1 = nn.Conv2d(n_feat*2, 1, 3, padding=1)
        self.out_p2 = nn.Conv2d(n_feat*2, 1, 3, padding=1)
        self.out_p3 = nn.Conv2d(n_feat*2, 1, 3, padding=1)
        self.out_p4 = nn.Conv2d(n_feat*2, 1, 3, padding=1)
        self.adapt = Adapter()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        
        
    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]



        x1_t = self.Translayer1_1(x1)#####channel=32
        x2_t = self.Translayer2_1(x2)
        x3_t = self.Translayer3_1(x3)
        x4_t = self.Translayer4_1(x4)
    
        x1_fe, x2_fe, x3_fe, x4_fe = self.fe(x1_t),self.fe(x2_t), self.fe(x3_t), self.fe(x4_t) #channel=64
        


        x4_fs = self.decoder4(self.fl4(self.adapt(self.expand(x4_fe))))  # c=128
        p1 = self.out_p1(x4_fs)

        p1_up = self.upsample(p1)

        x4_fs_up = self.upsample(x4_fs)

        x3_fs = self.decoder3(self.fl3(self.adapt(self.compress(torch.cat((x4_fs_up, x3_fe), 1)) )))
        p2 = self.out_p2(x3_fs)

        p2_up = self.upsample(p2)

        x3_fs_up = self.upsample(x3_fs)

        x2_fs = self.decoder2(self.adapt(self.fl2(self.compress(torch.cat((x3_fs_up, x2_fe), 1)) )))
        p3 = self.out_p3(x2_fs)

        p3_up = self.upsample(p3)

        x2_fs_up = self.upsample(x2_fs)

        x1_fs = self.decoder1(self.fl1(self.adapt(self.compress(torch.cat((x2_fs_up, x1_fe), 1)) )))
        p4 = self.out_p4(x1_fs)

        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')

        return [p1, p2, p3, p4]

