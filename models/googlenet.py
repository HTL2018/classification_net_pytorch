import torch
import torch.nn as nn

##################################
# Inception  建立Inception
##################################
class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        ##################################
        # input_channels : 输入通道数量, n1x1 : 1x1卷积核数量,也是1x1卷积层的输出通道数量
        #                              n3x3_reduce : 3x3卷积层前所用的1x1卷积核数量数量(,也是1x1卷积层的输出通道数量), n3x3 : 3x3卷积核数量,也是3x3卷积层的输出通道数量
        #                              n5x5_reduce, n5x5 : 3x3卷积层前所用的1x1卷积核数量数量(,也是1x1卷积层的输出通道数量), n3x3 : 3x3卷积核数量,也是3x3卷积层的输出通道数量
        #                              pool_proj : 最大池化后所用1x1卷积核数量数量(,也是1x1卷积层的输出通道数量)
        ##################################
        super().__init__()

        #1x1conv branch
        self.b1 = nn.Sequential(nn.Conv2d(input_channels, n1x1, kernel_size=1),nn.BatchNorm2d(n1x1),nn.ReLU(inplace=True))

        #1x1conv -> 3x3conv branch
        self.b2 = nn.Sequential(nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),nn.BatchNorm2d(n3x3_reduce),nn.ReLU(inplace=True),
                                nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),nn.BatchNorm2d(n3x3),nn.ReLU(inplace=True))

        #1x1conv -> 5x5conv branch
        #we use 2 3x3 conv filters stacked instead of 1 5x5 filters to obtain the same receptive field with fewer parameters
        self.b3 = nn.Sequential(nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),nn.BatchNorm2d(n5x5_reduce),nn.ReLU(inplace=True),
                                nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1),nn.BatchNorm2d(n5x5, n5x5),nn.ReLU(inplace=True),
                                nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),nn.BatchNorm2d(n5x5),nn.ReLU(inplace=True))

        #3x3pooling -> 1x1conv
        self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),
                                nn.Conv2d(input_channels, pool_proj, kernel_size=1),nn.BatchNorm2d(pool_proj),nn.ReLU(inplace=True))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

'''
##################################
# Check Inception  检验Inception
##################################
from tensorboardX import SummaryWriter 
import torch
from torch.autograd import Variable

writer = SummaryWriter(comment='-basic-inception')
sample_images = torch.randn(2, 64, 56, 56) # batche_size, input_channel, width, height
inception = Inception(64, 32, 32, 64, 16, 32, 64)

torch.onnx.export(inception, Variable(sample_images), 'inception.proto', verbose=True)  # 将model(inception)的结构和参数全部保存为 inception.proto
writer.add_onnx_graph('inception.proto')    # 将proto格式的文件转换为tensorboard中的graph
writer.close()  
'''
''' 
查看tensorboard中形成的网络时:
    cd path  (path为包含文件夹runs的路径)
    tensorboard --logdir runs
''' 

##################################
# GoogleNet  建立GoogleNet
##################################
class GoogleNet(nn.Module):

    def __init__(self, num_class=10):   # 针对不同的分类,此处可以设置, num_class=10 则返回10个类的概率
        ##################################
        # 关于类别设置的补充说明: (假设真正要分类的有10种类别)
        #   如果设置的 num_class > 10,则print打印后发现第10以后的数值全为负值,不影响网络的性能;
        #   如果设置的 num_class < 10,则进程直接中断,报错;
        ##################################
        super().__init__()
        self.prelayer = nn.Sequential(nn.Conv2d(3, 192, kernel_size=3, padding=1),nn.BatchNorm2d(192),nn.ReLU(inplace=True))

        #although we only use 1 conv layer as prelayer,
        #we still use name a3, b3.......
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        # (192, 64, 96, 128, 16, 32, 32) 分别对应: (input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        # 256 = (上一Inception模块中n1x1, n3x3, n5x5, pool_proj之和)即 64+128+32+32

        #"""In general, an Inception network is a network consisting of
        #modules of the above type stacked upon each other, with occasional 
        #max-pooling layers with stride 2 to halve the resolution of the 
        #grid"""
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
        # forward 进行网络构造时使用和上面一样的 最大池化层 ,故此处省略池化
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        #input feature size: 8*8*1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = nn.Linear(1024, num_class)
    
    def forward(self, x):
        output = self.prelayer(x)
        output = self.a3(output)
        output = self.b3(output)
        
        output = self.maxpool(output)

        output = self.a4(output)
        output = self.b4(output)
        output = self.c4(output)
        output = self.d4(output)
        output = self.e4(output)

        output = self.maxpool(output)

        output = self.a5(output)
        output = self.b5(output)

        #"""It was found that a move from fully connected layers to
        #average pooling improved the top-1 accuracy by about 0.6%, 
        #however the use of dropout remained essential even after 
        #removing the fully connected layers."""
        output = self.avgpool(output)
        output = self.dropout(output)
        output = output.view(output.size()[0], -1)  # 进入全连接层时,一般都进行此操作
        output = self.linear(output)

        return output
'''
##################################
# check GoogLeNet  检验GoogLeNet
##################################
from tensorboardX import SummaryWriter 
import torch
from torch.autograd import Variable

writer = SummaryWriter(comment="-googlenet")
sample_images = torch.randn(2, 3, 56, 56) # batche_size, input_channel, width, height
googlenet = GoogleNet()

torch.onnx.export(googlenet, Variable(sample_images), 'googlenet.proto', verbose=True)  # 将model(inception)的结构和参数全部保存为 inception.proto
writer.add_onnx_graph('googlenet.proto')    # 将proto格式的文件转换为tensorboard中的graph
writer.close()  
'''

def googlenet():
    return GoogleNet()