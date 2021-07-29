# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):  # 尺寸不发生变化
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):  # 输入通道数，卷积核个数，basic两个相同，故expansion=1
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):  # 输入通道数，卷积核个数，bottle两个不同，故expansion=4
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):  # basic时，num_channels = num_inchannels, bottleneck 则为4倍
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,  # 2 分支数，basic 块的类型, 4 块的数量, [32,64] 输入通道数
                 num_channels, fuse_method, multi_scale_output=True):  # [32,64] 输出通道数, sum 融合方法, true 多尺度输出
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)  # 2, basic, [4,4], [32,64]
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):  # 分支的数量和每个分支块数的数量一样, 3  [4,4,4]
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):  # 分支的数量和每个分支输出通道数的数量一样, 3  [32,64,128]
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):  # 分支的数量和每个分支输入出通道数的数量一样, 3  [32,64,128]
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):  # 0, basic, [4,4], [32,64]
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            # 步长不等于1，或者输入通道数不等于通道数x扩展，说明需要不同维度连接，即bottleneck结构，basic扩展为1，bottleneck为4
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                               momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],  # 残差跳连
                            num_channels[branch_index], stride, downsample))

        # 如果进行了维度不同的连接，则下一个输入通道数发生变化
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion

        for i in range(1, num_blocks[branch_index]):  # 跳连之外的卷积
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):  # 3， basic， [4,4,4], [32,64,128]
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))  # 0, basic, [4,4], [32,64]

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):  # 多尺度融合函数
        if self.num_branches == 1:  # 一个分支不融合，
            return None

        num_branches = self.num_branches  # 3
        num_inchannels = self.num_inchannels  # 【32，64,128】
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):  # [32,64,128]
            fuse_layer = []
            for j in range(num_branches):  # [32,64,128]
                if j > i:  # 上操作 i是右边,j是左边,i=j是水平不操作,j>i比如1,0 ,此时左边第一个，右边第零个，
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],  # 64
                                  num_inchannels[i],  # 32
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))  # 上采样,相当于resize
                elif j == i:  # j=i 不操作
                    fuse_layer.append(None)
                else:  # j<i 0, 1,下操作  # 当左右只差1的时候，卷积核为左通道数x3x3x右通道数，相差1以上的时候，前几个为左通道数x3x3x左通道数，左右一个为左通道数x3x3x右通道数
                    conv3x3s = []
                    for k in range(i - j):  # 2, 0 需要下两次
                        if k == i - j - 1:  # 左右只差一层，比如左边是1，右边是2，只需要一次3x3卷积
                            num_outchannels_conv3x3 = num_inchannels[i]  # 128
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],  # 32
                                          num_outchannels_conv3x3,  # 128
                                          3, 2, 1, bias=False),  # 3x3 s=2 p=1
                                nn.BatchNorm2d(num_outchannels_conv3x3)))
                        else:  # 左右差一层以上，比如左边j是0，右边是i2
                            num_outchannels_conv3x3 = num_inchannels[j]  # 32
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],  # 32
                                          num_outchannels_conv3x3,  # 32
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class PoseHigherResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64  # stem层处理后是128x128x64,此时是网络的正式开始，输入通道为64
        extra = cfg.MODEL.EXTRA
        super(PoseHigherResolutionNet, self).__init__()

        # 输入开始512x512x3

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        # 此时256x256x64
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        # 此时128x128x64

        ##
        # 第一阶段 好像只有第一阶段用了bottleneck,后边用的都是basic结构
        self.layer1 = self._make_layer(Bottleneck, 64, 4)  # 1/4之后的那层，4个bottleneck，第一阶段

        ##
        # 第二阶段
        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']  # 读取第二阶段的配置
        num_channels = self.stage2_cfg['NUM_CHANNELS']  # 两个通道，上边32下边64
        block = blocks_dict[self.stage2_cfg['BLOCK']]  # basic
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]  # basic 通道不发生变化
        # num_channels此时为[32,64]
        self.transition1 = self._make_transition_layer([256], num_channels)  # 256,[32,64] 256为第一阶段输出通道数，每个网络该阶段不发生变化
        # 此时上边为64x64x32 下边为32x32x64
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        ##
        # 第三阶段
        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        # 此时num_channels = [32,64,128]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        # 此时num_channels = [32,64,128],没变化因为后面都用的是basic,expansion=1
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)  # pre_stage_channels=[32,64] num_channels=[32,64,128]
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        # num_channels = [32,64,128,256]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        # pre_stage_channels 为该阶段的输出通道数，因为用的是basic，  所以num_channels = pre_stage_channels
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)

        # 此时只有一分支 特征图为64x64x32  # pre_stage_channels=[32,64,128,256],32
        self.final_layers = self._make_final_layers(cfg, pre_stage_channels[0])
        self.deconv_layers = self._make_deconv_layers(
            cfg, pre_stage_channels[0])  # 32

        self.num_deconvs = extra.DECONV.NUM_DECONVS
        self.deconv_config = cfg.MODEL.EXTRA.DECONV
        self.loss_config = cfg.LOSS

        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

    def _make_final_layers(self, cfg, input_channels):  # input_channels = 32
        #  维度标签，如果每个关节点标签都显示的话，则为关节点数量，否则为1
        dim_tag = cfg.MODEL.NUM_JOINTS if cfg.MODEL.TAG_PER_JOINT else 1  # 是否显示每个关节点的标签  14
        extra = cfg.MODEL.EXTRA

        final_layers = []
        # 下面这个不是太懂, output_channels相当于最终的结果类，有多少个需要的标签类
        output_channels = cfg.MODEL.NUM_JOINTS + dim_tag \
            if cfg.LOSS.WITH_AE_LOSS[0] else cfg.MODEL.NUM_JOINTS  # 如果这个为true的话是14+14否则为14

        # 相当于结果1/4对应的前面卷基层
        final_layers.append(nn.Conv2d(
            in_channels=input_channels,  # 32
            out_channels=output_channels,  # 14+14
            kernel_size=extra.FINAL_CONV_KERNEL,  # 1
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0  # 0
        ))

        # 反卷积层操作后的再次反卷积得到正常结果
        deconv_cfg = extra.DECONV
        for i in range(deconv_cfg.NUM_DECONVS):  # 1 表示就一个反卷积模块
            input_channels = deconv_cfg.NUM_CHANNELS[i]  # 32
            output_channels = cfg.MODEL.NUM_JOINTS + dim_tag \
                if cfg.LOSS.WITH_AE_LOSS[i + 1] else cfg.MODEL.NUM_JOINTS  # 14
            final_layers.append(nn.Conv2d(
                in_channels=input_channels,  # 32
                out_channels=output_channels,  # 14
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
            ))

        return nn.ModuleList(final_layers)

    def _make_deconv_layers(self, cfg, input_channels):  # 32
        dim_tag = cfg.MODEL.NUM_JOINTS if cfg.MODEL.TAG_PER_JOINT else 1  # 14
        extra = cfg.MODEL.EXTRA
        deconv_cfg = extra.DECONV

        deconv_layers = []
        for i in range(deconv_cfg.NUM_DECONVS):  # 1, 反卷积模块数量
            if deconv_cfg.CAT_OUTPUT[i]:  # 该模块是否拼接显示
                final_output_channels = cfg.MODEL.NUM_JOINTS + dim_tag \
                    if cfg.LOSS.WITH_AE_LOSS[i] else cfg.MODEL.NUM_JOINTS  # 28 或者 14
                input_channels += final_output_channels  # 32 + 28
            output_channels = deconv_cfg.NUM_CHANNELS[i]  # 32
            deconv_kernel, padding, output_padding = \
                self._get_deconv_cfg(deconv_cfg.KERNEL_SIZE[i])

            layers = []
            layers.append(nn.Sequential(  # 反卷积层
                nn.ConvTranspose2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=deconv_kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False),
                nn.BatchNorm2d(output_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ))
            # 反卷基层的4个basic块
            for _ in range(cfg.MODEL.EXTRA.DECONV.NUM_BASIC_BLOCKS):
                layers.append(nn.Sequential(
                    BasicBlock(output_channels, output_channels),
                ))
            deconv_layers.append(nn.Sequential(*layers))
            input_channels = output_channels

        return nn.ModuleList(deconv_layers)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):  # 转换层
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):  # 如 bootleneck ,64 ,4. 此处的planes相当于卷积核的个数
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:  # 如果步长不等于可能尺寸会发生变化，或者输入通道不等于基础倍数（64）x4
            downsample = nn.Sequential(  # 为bottleneck维度不同时的残差跳连结构
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))  # 第一个单元进行了残差跳连，维度不同，剩下三个维度相同

        self.inplanes = planes * block.expansion  # 更改此时过程的输入通道数,如bottleneck之前为64,此时为256,basic前后不发生变化。
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))  # bottleneck(256,64) basic(64,64)，输入通道数，卷积核的个数,

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,  # 创建一个阶段 num_inchannels = [32,64]
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']  # 该阶段hr单元数量 1,比如2阶段用了1，3阶段4，4阶段3
        num_branches = layer_config['NUM_BRANCHES']  # 单元几个分支 2
        num_blocks = layer_config['NUM_BLOCKS']  # 每个单元几个块 [4,4]
        num_channels = layer_config['NUM_CHANNELS']  # 不同分支的通道数,[32,64]
        block = blocks_dict[layer_config['BLOCK']]  # 块的类型，比如basic
        fuse_method = layer_config['FUSE_METHOD']  # 融合的方法

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            # 实际上只有最后一个hr模块的最后一个块没有多尺度输出，即别的输出结果是多个，最后的一个是1个
            if not multi_scale_output and i == num_modules - 1:  # 4个模块时  这处判断有问题,说是只对最后一个hr模块进行，但是实际上每个都进行了
                reset_multi_scale_output = False  # 不到最后一个单元，该判断一直成立,所以只有最后一个hr模块进行多尺度融合 1 2 3
            else:
                reset_multi_scale_output = True  # 4

            modules.append(
                HighResolutionModule(
                    num_branches,  # 2
                    block,  # basic
                    num_blocks,  # [4,4]
                    num_inchannels,  # [32,64] num_inchannels = num_channels*block.expansion, basic为1
                    num_channels,  # [32,64]
                    fuse_method,  # sum
                    reset_multi_scale_output)  # true
            )
            num_inchannels = modules[-1].get_num_inchannels()  # 得到最后一个hr模块的输出通道数，相当于下一个阶段的输入通道数

        return nn.Sequential(*modules), num_inchannels  # num_inchannels = [32,64]

    def forward(self, x):
        # x输入为512x512x3
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        # x为128x128x256
        # 下面开始转换层
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):  # 1
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        # 此时为x_list 64x64x32 32x32x64
        y_list = self.stage2(x_list)
        # 经过阶段2后，y_list 64x64x32 32x32x64
        # 第二阶段转换
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        # 此时为x_list 64x64x32 32x32x64 16x16x128
        y_list = self.stage3(x_list)
        # 此时为y_list 64x64x32 32x32x64 16x16x128

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        # 此时为x_list 64x64x32 32x32x64 16x16x128 8x8x256
        y_list = self.stage4(x_list)  #?
        # 此时为y_list 64x64x32 32x32x64 16x16x128 8x8x256

        final_outputs = []
        x = y_list[0]   # x相当于右边1/4对应的特征图 128x128x32
        y = self.final_layers[0](x)  # 上边final层的最终输出  1/4后边的那个特征图 32x32x28
        final_outputs.append(y)
        # 单张图片的实际格式x 128x128x32 y 128x128x28  实际上会有多张图片 x [n,32,128,128] y [n,28,128,128]
        for i in range(self.num_deconvs):  # 1 只有一个反卷基层
            if self.deconv_config.CAT_OUTPUT[i]:
                x = torch.cat((x, y), 1)  # C 操作，拼接上边两个特征图  ???

            # 32
            x = self.deconv_layers[i](x)   # 下边decov层的结果  # 256x256x32
            y = self.final_layers[i + 1](x)  # 下边decov层的结果再进行卷积得到最终的decov的最终输出  256x256x14
            final_outputs.append(y)

        return final_outputs

    def init_weights(self, pretrained='', verbose=True):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                        or self.pretrained_layers[0] is '*':
                    if name in parameters_names or name in buffers_names:
                        if verbose:
                            logger.info(
                                '=> init {} from {}'.format(name, pretrained)
                            )
                        need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)


def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHigherResolutionNet(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED, verbose=cfg.VERBOSE)

    return model


