"""mobilenetv2 in pytorch



[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen

    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
# Written by weiaicunzai
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

LEAKY_VALUE = 0.01  # 0.1
AF_TYPE = nn.LeakyReLU(LEAKY_VALUE, inplace=True)  # nn.ReLU6(inplace=True) #


class MaxPoolReplacement(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MaxPoolReplacement, self).__init__()
        # 1x1 convolution layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Applying 1x1 convolution instead of max pooling
        x = self.conv(x)
        return x

# Creating an instance of the MaxPoolReplacement module
# maxpool_replacement = MaxPoolReplacement(in_channels, out_channels)


class LinearBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            AF_TYPE,

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            AF_TYPE,

            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual


class MobileNetV2(nn.Module):

    def __init__(self, class_num=100, spec_layer_in=512):
        super().__init__()
        self.specail_in_c = spec_layer_in
        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32)
        )
        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)    # 2
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)    # 3
        self.stage4 = self._make_stage(1, 32, 64, 2, 6)    # 4
        self.stage5 = self._make_stage(4, 64, 96, 1, 6)    # 3
        self.stage6 = self._make_stage(1, 96, 160, 1, 6)   # 3
        # self.stage2 = self._make_stage(2, 16, 24, 2, 6)  # 2
        # self.stage3 = self._make_stage(3, 24, 32, 2, 6)  # 3
        # self.stage4 = self._make_stage(4, 32, 64, 2, 6)  # 4
        # self.stage5 = self._make_stage(3, 64, 96, 1, 6)  # 3
        # self.stage6 = self._make_stage(3, 96, 160, 1, 6)  # 3
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            AF_TYPE
        )

        self.conv2 = nn.Conv2d(1280, class_num, 1)
        self.conv5_large = self._make_other_layers(['M', 1024, 512, 1024])
        # self.conv5_large = self._make_other_layers(['M', 1024, 512, 1024, 512, 1024])

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):
        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)

    @staticmethod
    def conv_as_pool(input_c):
        return nn.Conv2d(input_c, input_c, kernel_size=3, stride=2, padding=1)

    @staticmethod
    def load_weights(model, weight_file="mobilenetv2_1.0-f2a8633.pth"):
        pretrained_dict = torch.load(weight_file)  # 预训练的MobilenetV2
        model_dict = model.state_dict()  # 读取自己的网络的结构参数
        print("{} \n---------pre_trained v.s. our_struct----------\n{}".format(pretrained_dict, model_dict))

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and (v.shape == model_dict[k].shape)}
        model_dict.update(pretrained_dict)  # 将与 pretrained_dict 中 layer_name 相同的参数更新为 pretrained_dict 的参数
        model.load_state_dict(model_dict)  # 加载更新后的参数
        return model

    def _make_other_layers(self, layer_cfg):
        layers = []

        # set the kernel size of the first conv block = 3
        kernel_size = 3
        for v in layer_cfg:
            print("[_make_other_layers]:{}".format(v))
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                print("in size {}, out size {}".format(self.specail_in_c, v))
                layers += conv_bn_leaky_for_last(self.specail_in_c, v, kernel_size)
                kernel_size = 1 if kernel_size == 3 else 3
                self.specail_in_c = v
        return nn.Sequential(*layers)


def mobilenet_v2_instantiated():
    return MobileNetV2()


def conv_bn_for_layer(use_relu6, in_channels, out_channels, kernel_size, return_module=False):
    if use_relu6 is True:
        layers = conv_bn_relu6(in_channels, out_channels, kernel_size, return_module=return_module)
    else:
        layers = conv_bn_leaky_for_last(in_channels, out_channels, kernel_size, return_module=return_module)
    return layers


def conv_bn_leaky_for_last(in_channels, out_channels, kernel_size, return_module=False):
    padding = int((kernel_size - 1) / 2)
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                        stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True)]
    if return_module:
        return nn.Sequential(*layers)
    else:
        return layers


def conv_bn_relu6(in_channels, out_channels, kernel_size, return_module=False):
    padding = int((kernel_size - 1) / 2)
    print("[conv_bn_relu6]: {} % 1 = {} ?= 0".format(out_channels, (out_channels % 1)))
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                        stride=1, padding=padding, bias=False),
              nn.BatchNorm2d(out_channels),
              nn.ReLU6(inplace=True)]
    if return_module:
        return nn.Sequential(*layers)
    else:
        return layers


def conv_bn_leaky(in_channels, out_channels, kernel_size, return_module=False):
    padding = int((kernel_size - 1) / 2)
    layers = [nn.LeakyReLU(LEAKY_VALUE, inplace=True),
              nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                        stride=1, padding=padding, bias=False),
              nn.BatchNorm2d(out_channels),
              nn.LeakyReLU(LEAKY_VALUE, inplace=True)]
    if return_module:
        return nn.Sequential(*layers)
    else:
        return layers

