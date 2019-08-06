import sys
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import models
from torch.optim.lr_scheduler import _LRScheduler

##################################
# get_network 选定用于训练的网络,确定是否使用 gpu 进行训练(默认使用)
##################################
def get_network(args, use_gpu=True):
    '''
    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    '''
    #elif args.net == 'googlenet':
    if args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    '''
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34 
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50 
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101 
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
   

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()  
     '''
    
    if use_gpu:
        net = net.cuda()

    return net

##################################
# get_training_dataloader 读取训练集的数据(包含了一些对图片的处理)
##################################
def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of imagenet training dataset
        std: std of imagenet training dataset
        path: path to imagenet training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: imagenet_training_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        #transforms.Scale(224,224),
        transforms.Resize((32,32)), # 将图片尺寸重新缩放为(3,32,32)如果太大在处理的时候,显存会溢出
        #transforms.RandomCrop(16, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    imagenet_training = torchvision.datasets.ImageFolder('/home/htl/data/htl/imagenet/train',transform=transform_train)
    imagenet_training_loader = DataLoader(imagenet_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return imagenet_training_loader

##################################
# get_test_dataloader 读取测试集的数据(包含了一些对图片的处理)
##################################
def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of imagenet test dataset
        std: std of imagenet test dataset
        path: path to imagenet test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: imagenet_test_loader:torch dataloader object
    """
    
    transform_test = transforms.Compose([
        transforms.Resize((32,32)), # 将图片尺寸重新缩放为(3,32,32)如果太大在处理的时候,显存会溢出
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    imagenet_testing = torchvision.datasets.ImageFolder('/home/htl/data/htl/imagenet/test',transform=transform_test)
    imagenet_test_loader = DataLoader(imagenet_testing, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return imagenet_test_loader
'''
##################################
# compute_mean_std 计算图像数据集的均值和方差
# (用到的imagenet的均值和方差是从imagenet的官网上找到的,不是计算得到的)
##################################
import numpy

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data
    
    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std
'''

##################################
# WarmUpLR 调整学习率,使用SGD优化器时才需要(使用Adam优化器时可以将其注释掉)
##################################
class WarmUpLR(_LRScheduler):     # SGD 使用Adam时可以注释掉,亦可以不注释
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self): 
        """we will use the first m batches, and set the learning rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]