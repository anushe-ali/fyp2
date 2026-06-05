"""
architectures.py
─────────────────
Exact model classes from the training notebook.

Audio  : ECAPA_TDNN (embedding_size=512)
Visual : IResNet    (model='res18', num_features=512)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ═══════════════════════════════════════════════════════════════
#  AUDIO MODEL — ECAPA-TDNN
# ═══════════════════════════════════════════════════════════════

class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super().__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        num_pad = math.floor(kernel_size / 2) * dilation
        convs, bns = [], []
        for _ in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size,
                                   dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns   = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3   = nn.BatchNorm1d(planes)
        self.relu  = nn.ReLU()
        self.width = width
        self.se    = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.bns[i](self.relu(self.convs[i](sp)))
            out = sp if i == 0 else torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)
        out = self.se(self.bn3(self.relu(self.conv3(out))))
        out += residual
        return out


class PreEmphasis(nn.Module):
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter',
            torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.pad(x, (1, 0), 'reflect')
        return F.conv1d(x, self.flipped_filter).squeeze(1)


class FbankAug(nn.Module):
    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        super().__init__()
        self.freq_mask_width = freq_mask_width
        self.time_mask_width = time_mask_width

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        chunk = x.unfold(dimension=dim, size=x.size(dim), step=x.size(dim))
        chunk = chunk.view(chunk.size(0), chunk.size(1), chunk.size(2), chunk.size(3))
        start, end = (self.freq_mask_width if dim == 1 else self.time_mask_width)
        value = torch.randint(start, end, (1,)).item()
        start_point = torch.randint(0, x.size(dim) - value, (1,)).item()
        zeros = torch.zeros_like(chunk)
        if dim == 1:
            chunk[:, start_point:start_point + value] = zeros[:, start_point:start_point + value]
        elif dim == 2:
            chunk[:, :, start_point:start_point + value] = zeros[:, :, start_point:start_point + value]
        return chunk.view(original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x


class ECAPA_TDNN(nn.Module):
    def __init__(self, C=1024, embedding_size=512):
        super().__init__()
        self.torchfbank = nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_fft=512, win_length=400,
                hop_length=160, f_min=20, f_max=7600,
                window_fn=torch.hamming_window, n_mels=80
            ),
        )
        self.specaug = FbankAug()

        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)

        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, embedding_size)
        self.bn6 = nn.BatchNorm1d(embedding_size)

    def forward(self, x, aug=False):
        with torch.no_grad():
            x = self.torchfbank(x) + 1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug:
                x = self.specaug(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)
        x  = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x  = self.relu(x)

        t = x.size()[-1]
        global_x = torch.cat((
            x,
            torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
            torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t),
        ), dim=1)

        w = self.attention(global_x)
        mu  = torch.sum(x * w, dim=2)
        sg  = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))
        x   = torch.cat((mu, sg), dim=1)
        x   = self.bn5(x)
        x   = self.fc6(x)
        x   = self.bn6(x)
        return x


# ═══════════════════════════════════════════════════════════════
#  VISUAL MODEL — IResNet (res18, num_features=512)
# ═══════════════════════════════════════════════════════════════

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1       = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1     = conv3x3(inplanes, planes)
        self.bn2       = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu     = nn.PReLU(planes)
        self.conv2     = conv3x3(planes, planes, stride)
        self.bn3       = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride    = stride

    def forward(self, x):
        identity = x
        out = self.conv1(self.bn1(x))
        out = self.prelu(self.bn2(out))
        out = self.bn3(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(self, block=IBasicBlock, model='res18', dropout=0,
                 num_features=512, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None):
        super().__init__()
        layers = [2, 2, 2, 2] if model == 'res18' else [3, 4, 14, 3]

        self.inplanes   = 64
        self.dilation   = 1
        self.groups     = groups
        self.base_width = width_per_group

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.bn2     = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc      = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample    = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers = [block(self.inplanes, planes, stride, downsample,
                        self.groups, self.base_width, previous_dilation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = self.features(x)
        return x
