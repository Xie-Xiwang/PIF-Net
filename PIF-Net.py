import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.functional import softmax
from functools import partial
from torchvision import models




nonlinearity = partial(F.relu, inplace=True)



class MCU(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=True, stride=1):  # 默认use_1x1conv=True
        super(MCU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=9, padding=4, stride=stride)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        # print('X',X.shape) # torch.Size([1, 3, 512, 512])
        # 未使用BN和Relu
        # Y = self.conv2(self.conv1(self.conv(X)))

        Y = F.relu(self.bn(self.conv(X)))
        # print('Y0', Y.shape) # torch.Size([1, 32, 510, 510])
        Y = F.relu(self.bn1(self.conv1(Y)))
        # print('Y1',Y.shape) # torch.Size([1, 32, 510, 510])
        Y = self.bn2(self.conv2(Y))
        # print('Y2',Y.shape)  # torch.Size([1, 32, 512, 512])
        if self.conv3:
            X = self.conv3(X)
            # print('X1*1',X.shape)

        return F.relu(Y+X)



class ThreeConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ThreeConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):

        return self.conv(input)

class RDB(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=True, stride=1):  # 默认use_1x1conv=True
        super(RDB, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,  padding=1,stride=stride)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3,  padding=1,stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,stride=stride)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        # print('X',X.shape) # torch.Size([1, 3, 512, 512])
        # 未使用BN和Relu
        # Y = self.conv2(self.conv1(self.conv(X)))

        Y = F.relu(self.bn(self.conv(X)))
        # print('Y', Y.shape) # torch.Size([1, 32, 510, 510])
        Y1 = F.relu(self.bn1(self.conv1(Y)))
        # print('Y1',Y1.shape) # torch.Size([1, 32, 510, 510])
        Y2 = self.bn2(self.conv2(Y1))
        # print('Y2',Y2.shape)  # torch.Size([1, 32, 512, 512])
        # o=self.conv3(Y)
        # o1 = self.conv3(Y1)
        # o2 = self.conv3(Y2)
        # FF=torch.cat([o,o1,o2],dim=1)
        # print("FF",FF.shape)

        if self.conv3:
            FF = torch.cat([Y, Y1, Y2], dim=1)
            # print("FF", FF.shape)
            # X = self.conv3(X)
            # print('X1*1',X.shape)

        return F.relu(FF)


# class RDB(nn.Module):
#     def __init__(self, in_ch, out_ch,use_1x1conv=True,stride=1):
#         super(RDB, self).__init__()
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch, ),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch),
#         )
#
#         if use_1x1conv:
#             self.conv3 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)
#         else:
#             self.conv3 = None
#
#     def forward(self, input):
#         Y=self.conv(input)
#         if self.conv3:
#             X = self.conv3(input)
#         return F.relu(Y+X)






class FirstConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FirstConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):

        return self.conv(input)

class resnet34_unet(nn.Module):
    def __init__(self, num_classes=3, out_channels=3, pretrained=True):
        super(resnet34_unet, self).__init__()


        resnet = models.resnet18(pretrained=pretrained)
        self.firstconv = FirstConv(3, 32)  # 为常规卷积
        self.conv1x1 = nn.Conv2d(32, 64, kernel_size=1, stride=1)
        self.firstmaxpool = nn.MaxPool2d(2)

        # 以下四层为resnet的四层
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # 以下为五个平行分支
        # 第一个分支
        self.conv1 = RDB(32, 32)
        # self.conv1_1x1 = nn.Conv2d(96, 32, kernel_size=1, stride=1)
        self.mid1_1 = nn.Conv2d(96, 32, kernel_size=3, padding=1, stride=1)
        self.conv1_1 = RDB(32, 32)
        self.pool1 = nn.MaxPool2d(2)

        # 第二个分支
        # self.M1 = MCU(3, 32)
        self.conv2 = RDB(64, 64)
        # self.conv2_1x1 = nn.Conv2d(224, 64, kernel_size=1, stride=1)
        self.mid2_1 = nn.Conv2d(224, 64, kernel_size=3, padding=1, stride=1)
        self.conv2_1 = RDB(64, 64)  # 只需要改变输入的通道数就行，即第一个参数
        self.pool2 = nn.MaxPool2d(2)

        # 第三个分支
        # self.M2 = MCU(3, 64)
        self.conv3 = RDB(128, 128)
        # self.conv3_1x1 = nn.Conv2d(448, 128, kernel_size=1, stride=1)
        self.mid3_1 = nn.Conv2d(448, 128, kernel_size=3, padding=1, stride=1)
        self.conv3_1 = RDB(128, 128)
        self.pool3 = nn.MaxPool2d(2)

        # 第四个分支
        # self.M3 = MCU(3, 256)
        self.conv4 = RDB(256, 256)
        # self.conv4_1x1 = nn.Conv2d(896, 256, kernel_size=1, stride=1)
        self.mid4_1 = nn.Conv2d(896, 256, kernel_size=3, padding=1, stride=1)
        self.conv4_1 = RDB(256, 256)
        self.pool4 = nn.MaxPool2d(2)

        # 第五个分支
        # self.M4 = MCU(3, 512)
        self.conv5 = RDB(512, 512)
        # self.conv5_1x1 = nn.Conv2d(1792, 512, kernel_size=1, stride=1)
        self.mid5_1 = nn.Conv2d(1792, 512, kernel_size=3, padding=1, stride=1)
        self.conv5_1 = RDB(512, 512)

        # 每个分支的最后一层卷积
        self.end5 = nn.Conv2d(1536, 512, kernel_size=3, padding=1, stride=1)
        self.end4 = nn.Conv2d(768, 256, kernel_size=3, padding=1, stride=1)
        self.end3 = nn.Conv2d(384, 128, kernel_size=3, padding=1, stride=1)
        self.end2 = nn.Conv2d(192, 64, kernel_size=3, padding=1, stride=1)
        self.end1 = nn.Conv2d(96, 32, kernel_size=3, padding=1, stride=1)

        # 输出阶段的三个卷积
        self.three_out = ThreeConv(992, 512)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        # Encoder, 主干网络
        x = self.firstconv(x)

        e1 = self.firstmaxpool(self.encoder1(self.conv1x1(x)))
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        print("vvvvv",x.shape,e1.shape,e2.shape,e3.shape,e4.shape)

        # 以下是四个平行分支
        # 第一个分支
        c1 = self.conv1(x)
        # c1_1x1=self.conv1_1x1(c1)
        m1_1 = self.mid1_1(c1)
        p1 = self.pool1(m1_1)
        out_1 = self.conv1_1(m1_1)
        # print(out_1.shape)


        # 第二个分支
        c2 = self.conv2(e1)
        ff1 = torch.cat([c2, p1], dim=1)
        # conv2_1x1=self.conv2_1x1(ff1)
        m2_1 = self.mid2_1(ff1)
        p2 = self.pool2(m2_1)
        out_2 = self.conv2_1(m2_1)

        # 第三个分支
        c3 = self.conv3(e2)
        ff2 = torch.cat([c3, p2], dim=1)
        # conv3_1x1 = self.conv3_1x1(ff2)
        m3_1 = self.mid3_1(ff2)
        p3 = self.pool3(m3_1)
        out_3 = self.conv3_1(m3_1)

        # 第四个分支
        c4 = self.conv4(e3)
        ff3 = torch.cat([c4, p3], dim=1)
        # conv4_1x1 = self.conv4_1x1(ff3)
        m4_1 = self.mid4_1(ff3)
        p4 = self.pool4(m4_1)
        out_4 = self.conv4_1(m4_1)

        # 第五个分支
        c5 = self.conv5(e4)
        ff4 = torch.cat([c5, p4], dim=1)
        # conv5_1x1 = self.conv5_1x1(ff4)
        m5_1 = self.mid5_1(ff4)
        out_5 = self.conv5_1(m5_1)

        # 对每个分支进行卷积核上采样操作，注意第一个分支不需要进行上采样
        up_out_5 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(self.end5(out_5)) #上采样16
        up_out_4 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(self.end4(out_4)) #上采样8
        up_out_3 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(self.end3(out_3)) #上采样4
        up_out_2 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(self.end2(out_2)) #上采样2
        up_out_1 = self.end1(out_1)

        # 对每个分支结果进行级联并使用卷积操作
        ff_out = torch.cat([up_out_5, up_out_4, up_out_3, up_out_2, up_out_1], dim=1)
        final = self.three_out(ff_out)


        return nn.Sigmoid()(final)


# 将每层的结果进行上采样
class resnet34_unet2(nn.Module):
    def __init__(self, num_classes=3, out_channels=3, pretrained=True):
        super(resnet34_unet2, self).__init__()


        resnet = models.resnet18(pretrained=pretrained)
        self.firstconv = FirstConv(3, 32)  # 为常规卷积
        self.conv1x1 = nn.Conv2d(32, 64, kernel_size=1, stride=1)
        self.firstmaxpool = nn.MaxPool2d(2)

        # 以下四层为resnet的四层
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # 以下为五个平行分支
        # 第一个分支
        self.conv1 = RDB(32, 32)
        # self.conv1_1x1 = nn.Conv2d(96, 32, kernel_size=1, stride=1)
        self.mid1_1 = nn.Conv2d(96, 32, kernel_size=3, padding=1, stride=1)
        self.conv1_1 = RDB(64, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)  # 1024,512

        # 第二个分支
        # self.M1 = MCU(3, 32)
        self.conv2 = RDB(64, 64)
        # self.conv2_1x1 = nn.Conv2d(224, 64, kernel_size=1, stride=1)
        self.mid2_1 = nn.Conv2d(224, 64, kernel_size=3, padding=1, stride=1)
        self.conv2_1 = RDB(128, 64)  # 只需要改变输入的通道数就行，即第一个参数
        self.pool2 = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # 1024,512

        # 第三个分支
        # self.M2 = MCU(3, 64)
        self.conv3 = RDB(128, 128)
        # self.conv3_1x1 = nn.Conv2d(448, 128, kernel_size=1, stride=1)
        self.mid3_1 = nn.Conv2d(448, 128, kernel_size=3, padding=1, stride=1)
        self.conv3_1 = RDB(256, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)  # 1024,512

        # 第四个分支
        # self.M3 = MCU(3, 256)
        self.conv4 = RDB(256, 256)
        # self.conv4_1x1 = nn.Conv2d(896, 256, kernel_size=1, stride=1)
        self.mid4_1 = nn.Conv2d(896, 256, kernel_size=3, padding=1, stride=1)
        self.conv4_1 = RDB(512, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)  # 1024,512

        # 第五个分支
        # self.M4 = MCU(3, 512)
        self.conv5 = RDB(512, 512)
        # self.conv5_1x1 = nn.Conv2d(1792, 512, kernel_size=1, stride=1)
        self.mid5_1 = nn.Conv2d(1792, 512, kernel_size=3, padding=1, stride=1)
        self.conv5_1 = RDB(512, 512)

        # 每个分支的最后一层卷积
        self.end5 = nn.Conv2d(1536, 512, kernel_size=3, padding=1, stride=1)
        self.end4 = nn.Conv2d(768, 256, kernel_size=3, padding=1, stride=1)
        self.end3 = nn.Conv2d(384, 128, kernel_size=3, padding=1, stride=1)
        self.end2 = nn.Conv2d(192, 64, kernel_size=3, padding=1, stride=1)
        self.end1 = nn.Conv2d(96, 32, kernel_size=3, padding=1, stride=1)

        # 输出阶段的三个卷积
        self.three_out = ThreeConv(992, 512)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        # Encoder, 主干网络
        x = self.firstconv(x)

        e1 = self.firstmaxpool(self.encoder1(self.conv1x1(x)))
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        print("vvvvv",x.shape,e1.shape,e2.shape,e3.shape,e4.shape)

        # 以下是四个平行分支
        # 第一个分支
        c1 = self.conv1(x)
        # c1_1x1=self.conv1_1x1(c1)
        m1_1 = self.mid1_1(c1)
        p1 = self.pool1(m1_1)
        # c1_1 = self.conv1_1(m1_1)
        # print(c1_1.shape)


        # 第二个分支
        c2 = self.conv2(e1)
        ff1 = torch.cat([c2, p1], dim=1)
        # conv2_1x1=self.conv2_1x1(ff1)
        m2_1 = self.mid2_1(ff1)
        p2 = self.pool2(m2_1)
        # c2_1 = self.conv2_1(m2_1)
        # out2=self.end2(c2_1)




        # 第三个分支
        c3 = self.conv3(e2)
        ff2 = torch.cat([c3, p2], dim=1)
        # conv3_1x1 = self.conv3_1x1(ff2)
        m3_1 = self.mid3_1(ff2)
        p3 = self.pool3(m3_1)
        # c3_1 = self.conv3_1(m3_1)
        # out3=self.end3(c3_1)

        # 第四个分支
        c4 = self.conv4(e3)
        ff3 = torch.cat([c4, p3], dim=1)
        # conv4_1x1 = self.conv4_1x1(ff3)
        m4_1 = self.mid4_1(ff3)
        p4 = self.pool4(m4_1)


        # 第五个分支
        c5 = self.conv5(e4)
        ff4 = torch.cat([c5, p4], dim=1)
        # conv5_1x1 = self.conv5_1x1(ff4)
        m5_1 = self.mid5_1(ff4)
        c5_1 = self.conv5_1(m5_1)
        out5=self.end5(c5_1)


        # 类似解码部分
        up_4 = self.up4(out5)
        cat4 = torch.cat([up_4, m4_1], dim=1)
        c4_1 = self.conv4_1(cat4)
        out4 = self.end4(c4_1)


        up_3 = self.up3(out4)
        cat3 = torch.cat([up_3, m3_1], dim=1)
        c3_1 = self.conv3_1(cat3)
        out3 = self.end3(c3_1)

        up_2 = self.up2(out3)
        cat2 = torch.cat([up_2, m2_1], dim=1)
        c2_1 = self.conv2_1(cat2)
        out2 = self.end2(c2_1)

        up_1 = self.up1(out2)
        cat1 = torch.cat([up_1, m1_1], dim=1)
        c1_1 = self.conv1_1(cat1)
        # out1 = self.end1(c1_1)




        # 对每个分支进行卷积核上采样操作，注意第一个分支不需要进行上采样
        up_out_5 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(out5) #上采样16
        up_out_4 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(out4) #上采样8
        up_out_3 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(out3) #上采样4
        up_out_2 = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(out2) #上采样2
        up_out_1 = self.end1(c1_1)


        # 对每个分支结果进行级联并使用卷积操作
        ff_out = torch.cat([up_out_5, up_out_4, up_out_3, up_out_2, up_out_1], dim=1)
        final = self.three_out(ff_out)


        return nn.Sigmoid()(final)

