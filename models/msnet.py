"""MixStyle w/ random shuffle
https://github.com/KaiyangZhou/mixstyle-release/blob/master/imcls/models/resnet_mixstyle.py
"""
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision
from torch.nn.functional import interpolate
import torch.nn.functional as F
from numbers import Number
# from torchvision.models.detection import FasterRCNN
from models.fasterRCNN_local import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from collections import OrderedDict
from math import log

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):

    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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

class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._out_features = 512 * block.expansion
        self.fc = nn.Identity()  # for DomainBed compatibility

        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        #---------head-------------
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # - low-level features
        x0 = self.maxpool(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x0, x1, x2, x3, x4

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.featuremaps(x)
        return x0, x1, x2, x3, x4

def init_pretrained_weights(model, model_url):
    pretrain_dict = model_zoo.load_url(model_url)
    model.load_state_dict(pretrain_dict, strict=False)

def resnet18(pretrained=True):
    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model

def resnet50(pretrained=True):
    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3])

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet50"])

    return model

def cus_sample(
    feat: torch.Tensor,
    mode=None,
    factors=None,
    *,
    interpolation="bilinear",
    align_corners=False,
) -> torch.Tensor:
    """
    :param feat: 输入特征
    :param mode: size/scale
    :param factors: shape list for mode=size or scale list for mode=scale
    :param interpolation:
    :param align_corners: 具体差异可见https://www.yuque.com/lart/idh721/ugwn46
    :return: the resized tensor
    """
    if mode is None:
        return feat
    else:
        if factors is None:
            raise ValueError(
                f"factors should be valid data when mode is not None, but it is {factors} now."
                f"feat.shape: {feat.shape}, mode: {mode}, interpolation: {interpolation}, align_corners: {align_corners}"
            )

    interp_cfg = {}
    if mode == "size":
        if isinstance(factors, Number):
            factors = (factors, factors)
        assert isinstance(factors, (list, tuple)) and len(factors) == 2
        factors = [int(x) for x in factors]
        if factors == list(feat.shape[2:]):
            return feat
        interp_cfg["size"] = factors
    elif mode == "scale":
        assert isinstance(factors, (int, float))
        if factors == 1:
            return feat
        recompute_scale_factor = None
        if isinstance(factors, float):
            recompute_scale_factor = False
        interp_cfg["scale_factor"] = factors
        interp_cfg["recompute_scale_factor"] = recompute_scale_factor
    else:
        raise NotImplementedError(f"mode can not be {mode}")

    if interpolation == "nearest":
        if align_corners is False:
            align_corners = None
        assert align_corners is None, (
            "align_corners option can only be set with the interpolating modes: "
            "linear | bilinear | bicubic | trilinear, so we will set it to None"
        )
    try:
        result = F.interpolate(feat, mode=interpolation, align_corners=align_corners, **interp_cfg)
    except NotImplementedError as e:
        print(
            f"shape: {feat.shape}\n"
            f"mode={mode}\n"
            f"factors={factors}\n"
            f"interpolation={interpolation}\n"
            f"align_corners={align_corners}"
        )
        raise e
    except Exception as e:
        raise e
    return result

class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out

class c0_down(nn.Module):
    def __init__(self, out_c):
        super(c0_down, self).__init__()
        self.down = ConvBNReLU(64, out_c, 3, 1, 1)
    def forward(self, x):
        x = self.down(x)
        return x

class c1_down(nn.Module):
    def __init__(self, out_c):
        super(c1_down, self).__init__()
        self.down = ConvBNReLU(256, out_c, 3, 1, 1)
    def forward(self, x):
        x = self.down(x)
        return x

class c2_down(nn.Module):
    def __init__(self, out_c):
        super(c2_down, self).__init__()
        self.down = ConvBNReLU(512, out_c, 3, 1, 1)
    def forward(self, x):
        x = self.down(x)
        return x

class c3_down(nn.Module):
    def __init__(self, out_c):
        super(c3_down, self).__init__()
        self.down = ConvBNReLU(1024, out_c, 3, 1, 1)
    def forward(self, x):
        x = self.down(x)
        return x

class c4_down(nn.Module):
    def __init__(self, out_c):
        super(c4_down, self).__init__()
        self.down = ConvBNReLU(2048, out_c, 3, 1, 1)
    def forward(self, x):
        x = self.down(x)
        return x

class SIU(nn.Module):
    def __init__(self, in_dim):
        super(SIU, self).__init__()
        t = int(abs((log(in_dim, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
        self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)

        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)

        self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_cat = ConvBNReLU(3*in_dim, in_dim, 3, 1, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, s, m, l):
        tgt_size = m.shape[2:]
        # print(tgt_size)
        # torch.Size([200, 200])
        # print('=================')
        # print(m.shape)
        # torch.Size([4, 64, 200, 200])

        l = self.conv_l_pre_down(l)
        # print('l.shape:', l.shape)
        #l.shape: torch.Size([4, 64, 300, 300])

        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        # print('l1.shape:', l.shape)
        # l1.shape: torch.Size([4, 64, 200, 200])
        l = self.conv_l_post_down(l)
        # print('l2.shape:', l.shape)
        # l2.shape: torch.Size([4, 64, 200, 200])


        m = self.conv_m(m)

        s = self.conv_s_pre_up(s)
        # print('s1.shape:', s.shape)
        # s1.shape: torch.Size([4, 64, 100, 100])
        s = cus_sample(s, mode="size", factors=m.shape[2:])
        # print('s2.shape:', s.shape)
        # s2.shape: torch.Size([4, 64, 200, 200])
        s = self.conv_s_post_up(s)


        attn = torch.cat((l, m, s), dim=1)
        attn = self.conv_cat(attn)  #256
        # lms = attn_l * l + attn_m * m + attn_s * s
        wei = self.avg_pool(attn)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        attn = attn * wei
        return attn

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

class C3Conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(C3Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        # self.bn = nn.BatchNorm2d(out_planes)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return x

class RF(nn.Module):
    # Revised from: Receptive Field Block Net for Accurate and Fast Object Detection, 2018, ECCV
    # GitHub: https://github.com/ruinmessi/RFBNet
    def __init__(self, in_channel, out_channel):
        super(RF, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1, stride=1, padding=0),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1, stride=1, padding=0),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1, stride=1, padding=0),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1, stride=1, padding=0),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, stride=1, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1, stride=1, padding=0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class adjustscale(nn.Module):
    def __init__(self, in_dim):
        super(adjustscale, self).__init__()
        t = int(abs((log(in_dim, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
        self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)

        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)

        self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_cat = ConvBNReLU(3*in_dim, in_dim, 3, 1, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, s, m, l):
        tgt_size = m.shape[2:]
        # print(tgt_size)
        # torch.Size([200, 200])
        # print('=================')
        # print(m.shape)
        # torch.Size([4, 64, 200, 200])

        l = self.conv_l_pre_down(l)
        # print('l.shape:', l.shape)
        #l.shape: torch.Size([4, 64, 300, 300])

        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        # print('l1.shape:', l.shape)
        # l1.shape: torch.Size([4, 64, 200, 200])
        l = self.conv_l_post_down(l)
        # print('l2.shape:', l.shape)
        # l2.shape: torch.Size([4, 64, 200, 200])

        # 尺度不变
        m = self.conv_m(m)


        s = self.conv_s_pre_up(s)
        # print('s1.shape:', s.shape)
        # s1.shape: torch.Size([4, 64, 100, 100])
        s = cus_sample(s, mode="size", factors=m.shape[2:])
        # print('s2.shape:', s.shape)
        # s2.shape: torch.Size([4, 64, 200, 200])
        s = self.conv_s_post_up(s)

        return s, m, l

class Mybackbone(nn.Module):
    def __init__(self, out_dim):
        super(Mybackbone, self).__init__()
        self.share_encoder = resnet50()

        self.c0_d = c0_down(out_c=out_dim)
        self.c1_d = c1_down(out_c=out_dim)
        self.c2_d = c2_down(out_c=out_dim)
        self.c3_d = c3_down(out_c=out_dim)
        self.c4_d = c4_down(out_c=out_dim)

        # --不同尺度相同层级信息融合--
        self.siu = SIU(out_dim)
        # self.siu1 = SIU(in_dim=256)
        # self.siu2 = SIU(in_dim=512)
        # self.siu3 = SIU(in_dim=1024)
        # self.siu4 = SIU(in_dim=2048)

        # 增大感受野
        self.rf2 = RF(out_dim*5, out_dim)

        self.rf3 = RF(out_dim*3, out_dim)

        self.rf5 = RF(out_dim*2, out_dim)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.c_conv_2 = nn.Conv2d(in_channels=out_dim * 2, out_channels=out_dim, kernel_size=3, stride=1, padding=1)
        self.c_conv_3 = nn.Conv2d(in_channels=out_dim * 3, out_channels=out_dim, kernel_size=3, stride=1, padding=1)
        self.c_conv_5 = nn.Conv2d(in_channels=out_dim * 5, out_channels=out_dim, kernel_size=3, stride=1, padding=1)
        self.scale_0 = adjustscale(in_dim=64)
        # self.global_avgpool = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        # print(x.shape)
        x_s = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_m = x
        x_l = F.interpolate(x, scale_factor=1.5, mode='bilinear')

        x_s_0, x_s_1, x_s_2, x_s_3, x_s_4 = self.share_encoder(x_s)
        x_m_0, x_m_1, x_m_2, x_m_3, x_m_4 = self.share_encoder(x_m)
        x_l_0, x_l_1, x_l_2, x_l_3, x_l_4 = self.share_encoder(x_l)

        s_feat_0, m_feat_0, l_feat_0 = self.scale_0(x_s_0, x_m_0, x_l_0)

        x_s_0, x_s_1, x_s_2, x_s_3, x_s_4 = self.c0_d(x_s_0), self.c1_d(x_s_1), self.c2_d(x_s_2), self.c3_d(x_s_3), self.c4_d(x_s_4)
        x_m_0, x_m_1, x_m_2, x_m_3, x_m_4 = self.c0_d(x_m_0), self.c1_d(x_m_1), self.c2_d(x_m_2), self.c3_d(x_m_3), self.c4_d(x_m_4)
        x_l_0, x_l_1, x_l_2, x_l_3, x_l_4 = self.c0_d(x_l_0), self.c1_d(x_l_1), self.c2_d(x_l_2), self.c3_d(x_l_3), self.c4_d(x_l_4)

        x1 = self.siu(x_s_0, x_m_0, x_l_0)  #128,200,200
        x2 = self.siu(x_s_1, x_m_1, x_l_1)  #128,200,200
        ##mid-level
        x3 = self.siu(x_s_2, x_m_2, x_l_2)  #128, 100,100
        ##high-level
        x4 = self.siu(x_s_3, x_m_3, x_l_3)   #128, 50,50
        x5 = self.siu(x_s_4, x_m_4, x_l_4)   #128, 25,25

        # x5 = self.rf5(x_A_5) #128,25,25
        # x4 = self.rf4(x_A_4) #128,50,50
        # x3 = self.rf3(x_A_3) #128,100,100
        # x2 = self.rf2(x_A_2) #128,200,200
        # x1 = self.rf1(x_A_1) #128,200,200
        f5 = x5
        f5_up = self.upsample_2(f5)
        f_45_cat = torch.cat((f5_up, x4), dim=1)
        f4 = self.rf5(f_45_cat)

        f_45_up = self.upsample_2(f_45_cat) #128+128
        f_345_cat = torch.cat((f_45_up, x3), dim=1)
        f3 = self.rf3(f_345_cat)

        f_345_cat_up = self.upsample_2(f_345_cat)
        f_12345_cat = torch.cat((f_345_cat_up, x2, x1), dim=1)
        f2 = self.rf2(f_12345_cat)

        features = OrderedDict()
        features['0'] = f2  
        features['1'] = f3  
        features['2'] = f4  
        features['3'] = s_feat_0
        features['4'] = m_feat_0
        features['5'] = l_feat_0

        return features
        # return features

def create_model(num_classes, pretrained=True, coco_model=False ):
    backbone = Mybackbone(out_dim=128)
    backbone.out_channels = 128
    anchor_sizes = ((32,), (64,), (128,),)
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0", "1", "2"],
            output_size=7,
            sampling_ratio=2
        )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=2,  # Num classes shoule be None when `box_predictor` is provided.
        min_size=512,
        max_size=512,
        rpn_anchor_generator=rpn_anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model

if __name__ == '__main__':
    from model_summary import summary

    model = create_model(num_classes=81, pretrained=True, coco_model=True)
    summary(model)
