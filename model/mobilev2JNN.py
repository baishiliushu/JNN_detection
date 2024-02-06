from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.mobilenetv2 import MobileNetV2
from model.mobilenetv2 import conv_bn_for_layer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from loss.loss import build_target, yolo_loss
from config import Config


class ReorgLayer(nn.Module):
    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        B, C, H, W = x.data.size()
        ws = self.stride
        hs = self.stride
        x = x.view(B, C, int(H / hs), hs, int(W / ws), ws).transpose(3, 4).contiguous()
        x = x.view(B, C, int(H / hs * W / ws), hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, int(H / hs), int(W / ws)).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, int(H / hs), int(W / ws))
        return x


class mobilev2JNN(nn.Module):

    def __init__(self):
        super(mobilev2JNN, self).__init__()
        self.post_layers_input_channels = 320
        self.use_relu6 = False
        backbone = MobileNetV2(spec_layer_in=self.post_layers_input_channels)
        # darknet19 = Darknet19()
        # backbone.load_weights()

        # JNN darknet backbone
        self.conv0 = nn.Sequential(backbone.pre)
        self.conv1 = nn.Sequential(backbone.stage1)

        self.maxpool1 = nn.MaxPool2d(2, 2)  #backbone.conv_as_pool(2) #  #

        self.joint1 = conv_bn_for_layer(self.use_relu6, 32, 16, kernel_size=3, return_module=True)

        self.conv2 = nn.Sequential(backbone.stage2)
        self.joint22 = conv_bn_for_layer(self.use_relu6, 48, 24, kernel_size=3, return_module=True)

        self.conv3 = nn.Sequential(backbone.stage3)
        self.joint2 = conv_bn_for_layer(self.use_relu6, 64, 32, kernel_size=3, return_module=True)
        self.conv4 = nn.Sequential(backbone.stage4)
        self.joint3 = conv_bn_for_layer(self.use_relu6, 128, 64, kernel_size=3, return_module=True)

        self.conv5 = nn.Sequential(backbone.stage5)

        self.conv6 = nn.Sequential(backbone.stage6)
        self.joint4 = conv_bn_for_layer(self.use_relu6, 320, 160, kernel_size=3, return_module=True)
        self.conv7 = nn.Sequential(backbone.stage7)
        self.joint5 = conv_bn_for_layer(self.use_relu6, 640, 512, kernel_size=3, return_module=True)

        # detection layers
        # self.joint6 = conv_bn_leaky(2048, 1024, kernel_size=3, return_module=True)
        self.downsampler = conv_bn_for_layer(self.use_relu6, self.post_layers_input_channels, 64, kernel_size=1, return_module=True)
        # when joint5 is 512 input
        self.reorg = ReorgLayer()
        self.conv55 = nn.Sequential(backbone.conv5_large)
        self.conv66 = nn.Sequential(conv_bn_for_layer(self.use_relu6, 1024, 1024, kernel_size=3, return_module=True),
                                    conv_bn_for_layer(self.use_relu6, 1024, 1024, kernel_size=3, return_module=True))

        self.conv77 = nn.Sequential(conv_bn_for_layer(self.use_relu6, 256 + 1024, 1024, kernel_size=3, return_module=True),
                                   nn.Conv2d(1024, 5 * len(Config.anchors), kernel_size=1))  # (coords + conf) * anchors

    def forward(self, query, target, target_boxes=None, num_boxes=None, training=False):
        output = self.conv0(target)
        output = self.conv1(output)
        output = self.maxpool1(output)
        qoutput = self.conv0(query)
        qoutput = self.conv1(qoutput)
        output = self.activate_cat((output, qoutput), 1)
        output = self.joint1(output)

        output = self.conv2(output)
        qoutput = self.conv2(qoutput)
        output = self.activate_cat((output, qoutput), 1)
        output = self.joint22(output)

        output = self.conv3(output)
        qoutput = self.conv3(qoutput)
        output = self.activate_cat((output, qoutput), 1)
        output = self.joint2(output)

        output = self.conv4(output)
        qoutput = self.conv4(qoutput)
        output = self.activate_cat((output, qoutput), 1)
        output = self.joint3(output)
        output = self.conv5(output)
        # qoutput = self.conv5(qoutput)
        output = self.conv6(output)
        # qoutput = self.conv6(qoutput)
        # shoutcut_up = torch.cat((output, qoutput), 1)
        # output = self.joint4(shoutcut_up)
        output = self.conv7(output)
        # output = torch.cat((output, shoutcut_up), 1)
        # output = self.joint5(output)

        shortcut = self.reorg(self.downsampler(output))
        output = self.conv55(output)
        # qoutput = self.conv5(qoutput)
        # output = torch.cat((output, qoutput), 1)
        # output = self.joint5(output)
        output = self.conv66(output)
        # qoutput = self.conv6(qoutput)
        # output = torch.cat((output, qoutput), 1)
        # output = self.joint6(output)

        output = self.activate_cat([shortcut, output], dim=1)
        output = self.conv77(output)

        # out -- tensor of shape (B, num_anchors * (5 + num_classes), H, W)
        bsize, _, h, w = output.size()

        # 5dim tensor represents (t_x, t_y, t_h, t_w, t_c)
        # reorganize the output tensor to shape (B, H * W * num_anchors, coords + conf)
        output = output.permute(0, 2, 3, 1).contiguous().view(bsize, h * w * len(Config.anchors), 5)

        # activate the output tensor
        # `sigmoid` for t_x, t_y, t_c; `exp` for t_h, t_w;
        # `softmax` for (class1_score, class2_score, ...)

        xy_pred = torch.sigmoid(output[:, :, 0:2])
        conf_pred = torch.sigmoid(output[:, :, 4:5])
        hw_pred = torch.exp(output[:, :, 2:4])
        delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)

        if training:
            output_variable = (delta_pred, conf_pred)
            output_data = [v.data for v in output_variable]
            gt_data = (target_boxes, num_boxes)
            target_data = build_target(output_data, gt_data, h, w)

            target_variable = [Variable(v) for v in target_data]
            box_loss, iou_loss = yolo_loss(output_variable, target_variable)

            return box_loss, iou_loss

        return delta_pred, conf_pred

    def activate_cat(self, input_s, add_activate=True, dim=1):
        if not self.use_relu6:
            activate_ = nn.LeakyReLU(0.1, inplace=True)
        else:
            activate_ = nn.ReLU6(inplace=True)
        input_s_after_activate = []
        for i in input_s:
            if add_activate:
                i = activate_(i)
            input_s_after_activate.append(i)
        output = torch.cat(input_s_after_activate, dim=dim)
        return output


if __name__ == '__main__':
    model = mobilev2JNN()
    im = np.random.randn(1, 3, 416, 416)
    im_variable = Variable(torch.from_numpy(im)).float()
    out = model(im_variable)
    delta_pred, conf_pred, class_pred = out
    print('delta_pred size:', delta_pred.size())
    print('conf_pred size:', conf_pred.size())
    print('class_pred size:', class_pred.size())



