"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: ctpn.py
@time: 2020/4/5 9:28

"""
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from models.vgg import *
from models.resnet import *
from models.mobilenet import *
from models.shufflenetv2 import *

class BasicConv(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 relu=True,
                 bn=True,
                 bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Im2col(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(Im2col, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        height = x.shape[2]
        x = F.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)
        x = x.reshape((x.shape[0], x.shape[1], height, -1))
        return x

class BLSTM(nn.Module):

    def __init__(self, channel, hidden_unit, bidirectional=True):
        super(BLSTM, self).__init__()
        self.lstm = nn.LSTM(channel, hidden_unit, bidirectional=bidirectional)

    def forward(self, x):
        """
        WARNING: The batch size of x must be 1.
        """
        x = x.transpose(1, 3)
        recurrent, _ = self.lstm(x[0])
        recurrent = recurrent[np.newaxis, :, :, :]
        recurrent = recurrent.transpose(1, 3)
        return recurrent



    
class CTPN_Model(nn.Module):
    def __init__(self,base_model,pretrained):
        super(CTPN_Model, self).__init__()
        self.cnn = nn.Sequential()
        if('vgg' in base_model):
            self.cnn.add_module(base_model, globals().get(base_model)(pretrained=pretrained))
        elif('mobile' in base_model):
            self.cnn.add_module(base_model, globals().get(base_model)(pretrained=pretrained))
        elif('shuffle' in base_model):
            self.cnn.add_module(base_model, globals().get(base_model)(pretrained=pretrained))
        elif('resnet' in base_model):
            self.cnn.add_module(base_model,globals().get(base_model)(pretrained=pretrained,model_name=base_model))
        else:
            print('not support this base model')
        
#         self.rnn = nn.Sequential()
#         self.rnn.add_module('im2col', Im2col((3, 3), (1, 1), (1, 1)))
#         self.rnn.add_module('blstm', BLSTM(3 * 3 * 512, 128))
#         self.FC = nn.Conv2d(256, 512, 1)

        self.rpn = BasicConv(512, 512, 3,1,1,bn=False)
        self.brnn = nn.GRU(512,128, bidirectional=True, batch_first=True)
        self.lstm_fc = BasicConv(256, 512,1,1,relu=True, bn=False)
        

#######################################################################################################       
        self.vertical_coordinate = nn.Conv2d(512, 4 * 10, 1)
        self.score = nn.Conv2d(512, 2 * 10, 1)
        self.side_refinement = nn.Conv2d(512, 10, 1)

    def forward(self, x, val=False):
        
        x = self.cnn(x)
        
#         #########################################################        
#         b ,_ ,_ ,_ = x.shape 
#         batch_features = []
#         for i in range(b):
#             feature = self.rnn(x[i].unsqueeze(0))
#             batch_features.append(feature.squeeze(0))

#         x = torch.stack(batch_features,0)
#         x = self.FC(x)
#         x = F.relu(x, inplace=True)
#         ###########################################################
    
        ###########################
        x = self.rpn(x)

        x1 = x.permute(0,2,3,1).contiguous()  # channels last
        b = x1.size()  # batch_size, h, w, c
        x1 = x1.view(b[0]*b[1], b[2], b[3])

        x2, _ = self.brnn(x1)

        xsz = x.size()
        x3 = x2.view(xsz[0], xsz[2], xsz[3], 256)  # torch.Size([4, 20, 20, 256])

        x3 = x3.permute(0,3,1,2).contiguous()  # channels first
        x3 = self.lstm_fc(x3)
        x = x3
        ############################

        
        
        vertical_pred = self.vertical_coordinate(x)
        score = self.score(x)
        side_refinement = self.side_refinement(x)
        
        return score,vertical_pred,side_refinement
