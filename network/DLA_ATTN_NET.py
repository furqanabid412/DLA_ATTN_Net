import numpy as np
import torch
from torch import nn
import math
import torch.nn.functional as F
__all__ = ["DLA_ATTN_NET"]

BatchNorm = nn.BatchNorm2d
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(out_channels)
        self.stride = stride
        self.project = None
        if in_channels!=out_channels:
            self.project = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.project != None:
            residual = self.project(residual)
        out += residual
        out = self.relu(out)

        return out

class Deconv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=(1,2),stride=(1,2),padding=0)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Feature_Aggregator(nn.Module):
    def __init__(self,in_channels_1,in_channels_2,out_channels):
        super().__init__()
        self.deconv = Deconv(in_channels_2,out_channels)
        self.block_1 = BasicBlock(in_channels_1+in_channels_2,out_channels)
        self.block_2 = BasicBlock(out_channels,out_channels)

    def forward(self,x1,x2):
        x2 = self.deconv(x2)
        x1 = torch.cat([x1,x2],1)
        x1 = self.block_1(x1)
        x1 = self.block_2(x1)
        return x1

class DownSample(nn.Module):
    def __init__(self,in_channels,out_channels,stride=2, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=(1,2), padding=dilation, bias=False, dilation=dilation)
        self.bn1 = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1, padding=dilation,bias=False, dilation=dilation)
        self.bn2 = BatchNorm(out_channels)
        self.project = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=(1,2),padding=1)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        # x : B,C,H,W
        out = self.conv1(x) # outputs : B,C,H/2,W/2
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.project(residual)
        out += residual
        out = self.relu(out)
        return out

class Feature_Extractor(nn.Module):
    def __init__(self,in_channels,out_channels,num_blocks=6,down_sample_input=False):
        super().__init__()
        self.down_sample = None
        self.down_sample_input = down_sample_input
        if down_sample_input:
            self.down_sample = DownSample(in_channels,out_channels)

        blocks_modules = []
        for i in range(num_blocks):
            if i == 0 and not down_sample_input:
                blocks_modules.append(BasicBlock(in_channels,out_channels))
            else:
                blocks_modules.append(BasicBlock(out_channels,out_channels))
        self.blocks = nn.Sequential(*blocks_modules)

    def forward(self,x):
        if self.down_sample_input:
            x = self.down_sample(x)
        x = self.blocks(x)
        return x


class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        #self.bn_qk = nn.BatchNorm2d(groups)
        #self.bn_qr = nn.BatchNorm2d(groups)
        #self.bn_kr = nn.BatchNorm2d(groups)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        #stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)

        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


class axial_attn(nn.Module):
    def __init__(self,channel,w_kernel,h_kernel=32,stride=1):
        super().__init__()
        self.height_block = AxialAttention(in_planes=channel, out_planes=channel, groups=2, kernel_size=h_kernel)
        self.width_block = AxialAttention(in_planes=channel, out_planes=channel, groups=2, kernel_size=w_kernel, stride=stride,width=True)
        # self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
    def forward(self,x):
        x = x.clone()
        out = self.height_block(x)
        out = self.width_block(out)
        # out = self.relu1(out)
        out += x
        # out = out.clone()
        out = self.relu2(out)
        return out

class DLA_ATTN_NET(nn.Module):
    def __init__(self,num_inputs, channels,num_outputs):
        super().__init__()
        # num_inputs=64
        # channels=[64,128,128]
        # num_outputs = 4

        # self.input_bn = BatchNorm(num_inputs)
        self.extract_1a = Feature_Extractor(num_inputs,channels[0]) # 64,64
        self.extract_2a = Feature_Extractor(channels[0],channels[1],down_sample_input=True) # 64,128
        self.extract_3a = Feature_Extractor(channels[1],channels[2],down_sample_input=True) # 128,128
        self.aggregate_1b = Feature_Aggregator(channels[0],channels[1],channels[1]) # 64,128,128
        self.aggregate_1c = Feature_Aggregator(channels[1],channels[2],channels[2]) # 128,128,128
        self.aggregate_2b = Feature_Aggregator(channels[1],channels[2],channels[2]) # 128,128,128
        self.conv_1x1 = nn.Conv2d(channels[2],num_outputs,kernel_size=1,stride=1) # 128,4
        self.ax_attn_block1 = axial_attn(channel=channels[2], w_kernel=256, h_kernel=32, stride=1)
        self.ax_attn_block2 = axial_attn(channel=channels[1], w_kernel=512, h_kernel=32, stride=1)
        # self.ax_attn_block3 = axial_attn(channel=channels[0], w_kernel=1024, h_kernel=32, stride=1)

    def forward(self,x):
        # batch normalizing the input
        # x=self.input_bn(x)

        # Network
        x_1a = self.extract_1a(x.type(torch.float32))
        # x_1a = self.ax_attn_block3(x_1a)
        x_2a = self.extract_2a(x_1a)
        x_2a = self.ax_attn_block2(x_2a)
        x_3a = self.extract_3a(x_2a)
        x_3a = self.ax_attn_block1(x_3a)

        x_1b = self.aggregate_1b(x_1a,x_2a)
        x_2b = self.aggregate_2b(x_2a,x_3a)
        x_1c = self.aggregate_1c(x_1b,x_2b)
        out = self.conv_1x1(x_1c)
        return out
