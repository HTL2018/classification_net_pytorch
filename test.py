""" test neuron network performace print top1 and top5 err on test dataset of a model
        检测神经网络的性能,输出模型在测试集上的top1和top5错误率
    在使用 test.py 文件时,必须传入-weights 参数 (即训练好的模型文件的路径)
        例如:CUDA_VISIBLE_DEVICES=0 python test.py -net googlenet -b 6 -weights /home/htl/data/htl/temp/googlenet/2019-08-05T19:24:55.905332/googlenet-125-best.pth
"""

import argparse
from utils import get_network, get_test_dataloader
from conf import global_settings
import torch
from torch.autograd import Variable

'''
from matplotlib import pyplot as plt


import torchvision.transforms as transforms
from torch.utils.data import DataLoader
'''

if __name__ == '__main__':

    ############### 解析命令行参数和选项 ###############
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    args = parser.parse_args()

    net = get_network(args)

    ############### 读取测试集数据 ###############
    imagenet_test_loader = get_test_dataloader(
        global_settings.IMAGENET_TRAIN_MEAN,
        global_settings.IMAGENET_TRAIN_MEAN,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    ############### 加载指定模型的权重文件进行测试 ###############
    net.load_state_dict(torch.load(args.weights), args.gpu)
    #print(net)  # 可以输出网络的详细信息,参数,各个网络层的设置
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    # len(imagenet_test_loader.dataset):1300
    # len(imagenet_test_loader):测试集总数/bitchsize (此处:当bitchsize为64时,1300/64=21)
    # 迭代一次,即一个 n_iter ,处理1个bitch的图片
    for n_iter, (image, label) in enumerate(imagenet_test_loader):                                  
        print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(imagenet_test_loader)))   
        image = Variable(image).cuda()
        label = Variable(label).cuda()
        output = net(image)     # output.size为:(bitchsize,10)(因为数据集是10分类,是10个概率值)
        _, pred = output.topk(5, 1, largest=True, sorted=True)  # _.size为:(bitchsize,5); _是对每张图片预测的针对10个类别的10个概率的前5个最大值
                                                                # 注意:此处的 1 表示按行取值
                                                                # pred为相应概率在该行中的索引,即模型对图片预测的类别
        label = label.view(label.size(0), -1).expand_as(pred)   # 将图片的真实标签扩充变换,使其预测的格式一样 label.size():torch.Size([bitchsize])   label.size(0):bitchsize
        correct = pred.eq(label).float()

        #compute top 5
        correct_5 += correct[:, :5].sum()

        #compute top1 
        correct_1 += correct[:, :1].sum()

    ############### 输出:Top 1 err;Top 5 err;网络中参数的总数量 ###############
    print()
    print("Top 1 err: ", 1 - correct_1 / len(imagenet_test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(imagenet_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters()))) # numel()函数：返回数组中元素的个数