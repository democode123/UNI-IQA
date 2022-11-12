import torch.nn as nn
from torchvision import models
import torch
import torch.nn.functional as F
from SCNN import SCNN
# from CAM import CAM
import argparse
import numpy as np
from einops import rearrange
from copy import deepcopy
import cv2
import time
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

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
        return x

class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class Adaptive_Fusion(nn.Module):

    def __init__(self, channel):
        super(Adaptive_Fusion, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)

    def forward(self, x1, x2, x3): #torch.Size([5, 32, 7, 7])#torch.Size([5, 32, 14, 14])#torch.Size([5, 32, 28, 28])
        x1_1 = x1
        # x2_1___ = self.conv_upsample1(self.upsample(x1))
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2 #torch.Size([5, 32, 14, 14])
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3 #28, 28

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        return x3_2

class UNI_IQA(nn.Module):
    def __init__(self, config):

        """Declare all needed layers."""
        nn.Module.__init__(self)
        self.config = config

        # Convolution and pooling layers of resnet34.
        self.backbone = models.resnet34(pretrained=True)
        self.backbone2 = models.resnet34(pretrained=True)
        scnn = SCNN()
        scnn = torch.nn.DataParallel(scnn).cuda()
        scnn.load_state_dict(torch.load(config.scnn_root))
        self.sfeatures = scnn.module.features

        # Linear classifier.
        self.fc_sci1 = torch.nn.Linear(512, 1)
        self.fc_scnn = torch.nn.Linear(128, 1)
        self.fc_text = torch.nn.Linear(512, 1)
        self.fc_fusion = torch.nn.Linear(384, 1)

        self.pooling_sci1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pooling_ni1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pooling_text = nn.AdaptiveAvgPool2d((1, 1))
        self.pooling_fusion = nn.AdaptiveAvgPool2d((2, 2))

        self.conv1 = nn.Conv2d(128, 1, 1, 1, 0)
        # self.downconv = nn.Conv2d(512, 128, 1, 1, 0)

        self.up_pic = torch.nn.Upsample(size=None, scale_factor=16, mode='nearest', align_corners=None)
        self.up_text = torch.nn.Upsample(size=None, scale_factor=16, mode='nearest', align_corners=None)

        self.NS_thresh = nn.Parameter(torch.zeros(2))
        self.S_thresh = nn.Parameter(torch.zeros(4))

        #Components of AF module
        self.atten_channel_1=ChannelAttention(64)
        self.atten_channel_2=ChannelAttention(64)
        self.atten_channel_3=ChannelAttention(128)
        self.atten_channel_4=ChannelAttention(256)
        self.atten_channel_5 = ChannelAttention(512)

        self.atten_spatial_1=SpatialAttention()
        self.atten_spatial_2=SpatialAttention()
        self.atten_spatial_3=SpatialAttention()
        self.atten_spatial_4=SpatialAttention()
        self.atten_spatial_5 = SpatialAttention()

        self.rfb0 = GCM(128, 32)
        self.rfb1 = GCM(256, 32)
        self.rfb2 = GCM(512, 32)
        self.af = Adaptive_Fusion(32)

        # if self.config.CAM:
            # Freeze all previous layers.
            # for param in self.sfeatures.parameters():
            #     param.requires_grad = False
        # Initialize the fc layers.
        nn.init.kaiming_normal_(self.fc_sci1.weight.data)
        if self.fc_sci1.bias is not None:
            nn.init.constant_(self.fc_sci1.bias.data, val=0)
        nn.init.kaiming_normal_(self.fc_scnn.weight.data)
        if self.fc_scnn.bias is not None:
            nn.init.constant_(self.fc_scnn.bias.data, val=0)
        nn.init.kaiming_normal_(self.fc_text.weight.data)
        if self.fc_text.bias is not None:
            nn.init.constant_(self.fc_text.bias.data, val=0)
        nn.init.kaiming_normal_(self.fc_fusion.weight.data)
        if self.fc_fusion.bias is not None:
            nn.init.constant_(self.fc_fusion.bias.data, val=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def forward(self, ni, sci):
        """Forward pass of the network.
        """
        N = ni.size()[0]
        ## scnn
        ni1 = self.sfeatures(ni)
        ni1 = self.pooling_ni1(ni1)
        ni1 = ni1.view(N, 128)
        ni1 = F.normalize(ni1)
        head_ni1 = self.fc_scnn(ni1)

        if self.config.CAM:
            sci_a = self.sfeatures(sci)  # torch.Size([B, 128, 14, 14])
            att = self.up_text(self.conv1(sci_a))
            # att = F.normalize(att,p=2)
            # print(att)
            pic_att = att
            pic_att[pic_att < torch.median(pic_att)] = 0
            # print('pic_att_sum', torch.sum(pic_att))
            pic=torch.mul(pic_att, sci)

            text_att = att
            text_att[text_att >= torch.median(text_att)] = 0
            # print('text_att_sum', torch.sum(text_att))
            text=torch.mul(text_att, sci)

            #Freeze all previous layers.
            for param in self.sfeatures.parameters():
                param.requires_grad = False

        ## pic_scnn
        for param in self.sfeatures.parameters():
            param.requires_grad = True

        pic = self.sfeatures(pic)
        pic = self.pooling_ni1(pic)
        pic = pic.view(N, 128)
        pic = F.normalize(pic)
        head_pic = self.fc_scnn(pic)

        ## text_resnet
        text = self.backbone2.conv1(text)
        text = self.backbone2.bn1(text)
        text = self.backbone2.relu(text)
        text0 = self.backbone2.maxpool(text)

        text1 = self.backbone2.layer1(text0)
        text2 = self.backbone2.layer2(text1)
        text3 = self.backbone2.layer3(text2)
        text4 = self.backbone2.layer4(text3)

        text = self.pooling_text(text4)
        text = text.view(N, 512)
        text = F.normalize(text)
        head_text = self.fc_text(text)

        ## resnet
        sci = self.backbone.conv1(sci)
        sci = self.backbone.bn1(sci)
        sci = self.backbone.relu(sci)
        sci0 = self.backbone.maxpool(sci)

        temp = text0.mul(self.atten_channel_1(text0))
        temp = temp.mul(self.atten_spatial_1(temp))
        sci0=sci0+temp
        sci1 = self.backbone.layer1(sci0)

        temp = text1.mul(self.atten_channel_2(text1))
        temp = temp.mul(self.atten_spatial_2(temp))
        sci1=sci1+temp
        sci2 = self.backbone.layer2(sci1)

        temp = text2.mul(self.atten_channel_3(text2))
        temp = temp.mul(self.atten_spatial_3(temp))
        sci2=sci2+temp
        sci3 = self.backbone.layer3(sci2) #torch.Size([5, 256, 14, 14])

        temp = text3.mul(self.atten_channel_4(text3))
        temp = temp.mul(self.atten_spatial_4(temp))
        sci3=sci3+temp
        sci4 = self.backbone.layer4(sci3) #torch.Size([2, 512, 7, 7])

        temp = text4.mul(self.atten_channel_5(text4))
        temp = temp.mul(self.atten_spatial_5(temp))
        sci5=sci4+temp #torch.Size([2, 512, 7, 7])

        sci = self.pooling_sci1(sci4)
        sci = sci.view(N, 512)
        sci = F.normalize(sci)
        head_sci = self.fc_sci1(sci)

        x0 = self.rfb0(sci2) #torch.Size([5, 32, 28, 28])
        x1 = self.rfb1(sci3) #torch.Size([5, 32, 14, 14])
        x2 = self.rfb2(sci5) #torch.Size([5, 32, 7, 7])
        fu = self.af(x2, x1, x0)

        fu = self.pooling_fusion(fu)
        fu = fu.view(N, -1)
        fu = F.normalize(fu)
        head_fusion = self.fc_fusion(fu)

        S_score_fusion = self.S_thresh[0]*head_sci+self.S_thresh[1]*head_pic+self.S_thresh[2]*head_text+self.S_thresh[3]*head_fusion
        # final_score = self.NS_thresh[0]*S_score_fusion + self.NS_thresh[1]*head_ni1

        return S_score_fusion, head_ni1

def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scnn_root", type=str, default='saved_weights/scnn.pkl')
    parser.add_argument("--CAM", type=bool, default=True)
    return parser.parse_args()


