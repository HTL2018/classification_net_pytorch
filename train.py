import argparse
from conf import global_settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR
import torch.nn as nn
import torch.optim as optim
import os
import torch

from torch.autograd import Variable

from tensorboardX import SummaryWriter

##################################
# 训练设置
##################################
def train(epoch):

    net.train() # 注释掉此句,程序照常运行
    train_loss = 0
    train_correct = 0.0

    for batch_index, (images, labels) in enumerate(imagenet_training_loader):
        if epoch <= args.warm:  # SGD
            warmup_scheduler.step()  # SGD

        images = Variable(images)
        labels = Variable(labels)

        labels = labels.cuda()  # 转换为gpu能够处理的数据 
        images = images.cuda()

        optimizer.zero_grad()   # 梯度置零，也就是把loss关于weight的导数变成0.
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()   # 进行单次优化,参数更新(完成一个batch的训练之后,根据上面计算的梯度,对参数进行更新优化)

    #####################################################################################################    
        n_iter = (epoch - 1) * len(imagenet_training_loader) + batch_index + 1                          #
        ''' 迭代次数 n_iter 计算的说明:(13000是针对我们的数据集而言的,共13000张图片)
        # 训练期间总迭代次数: 13000*0.9=11700
        # 所以len(imagenet_training_loader) = 11700 / bitch_size 
        # 最后 +1 是因为batch_index是从0开始的
        '''

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():    # 两个参数分别是'weight'和'bias',
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tTrain_Loss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),     # len(images)就是bitch_size,此处是指当前图片在当前batch中是第几张
            total_samples=len(imagenet_training_loader.dataset)))   # len(imagenet_training_loader.dataset)是总的用来训练的图片的总数(11700)

        ############### 记录每次迭代的训练损失 ###############
        writer.add_scalar('Train/loss', loss.item(), n_iter)
        ############### 记录网络每层的weight和bias ###############
        for name, param in net.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]
            writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    ############### 计算并输出在训练集上的准确率和损失,一个Epoch输出一次 ###############
        train_loss += loss.item()    # 经过此操作将tensor转化为numpy格式,转换为numpy格式主要是为了计算当前损失之后进行输出
        _, preds = outputs.max(1)   # .max(1)返回行的最大值,.max(0)返回列的最大值   _代表最大的值,preds表示最大的值所在的索引
        train_correct += preds.eq(labels).sum()

    print() 
    print('Train set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        train_loss / len(imagenet_training_loader.dataset),
        train_correct.float() / len(imagenet_training_loader.dataset)
    ))
    print()                                                                                            #
    ################################此代码块是主要为了保存显示一些关键参数####################################

##################################
# 在测试集上对模型进行评估
##################################
def eval_training(epoch):

    net.eval() # 注释掉此句,程序照常运行 
    test_loss = 0.0 
    correct = 0.0

    for images, labels in imagenet_test_loader:   
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()  # labels.size() = bitch_size 

        outputs = net(images)   # outputs.size()为torch.Size([32, 10]),其中32是设定的batchsize,10分别对应10类的预测概率
        loss = loss_function(outputs, labels)   # 此处的loss是tensor
        test_loss += loss.item()    # 经过此操作将tensor转化为numpy格式,转换为numpy格式主要是为了计算当前损失之后进行输出
        _, preds = outputs.max(1)   # .max(1)返回行的最大值,.max(0)返回列的最大值   _代表最大的值,preds表示最大的值所在的索引
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(imagenet_test_loader.dataset),
        correct.float() / len(imagenet_test_loader.dataset)
    ))
    print()

    ############### add informations to tensorboard:记录测试集上的损失和准确率到tensorboard中 ###############
    writer.add_scalar('Test/Average loss', test_loss / len(imagenet_test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(imagenet_test_loader.dataset), epoch)

    return correct.float() / len(imagenet_test_loader.dataset)

##################################
# 执行 train.py文件后  由此开始执行
##################################
if __name__ == '__main__':
    
    ############### 解析命令行参数和选项 ###############
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()

    net = get_network(args, use_gpu=args.gpu)   # 调用utils.py中的方法,形成网络


    ############### 读取训练集和测试集数据 ###############
    imagenet_training_loader = get_training_dataloader(
        global_settings.IMAGENET_TRAIN_MEAN,   
        global_settings.IMAGENET_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )

    imagenet_test_loader = get_test_dataloader(
        global_settings.IMAGENET_TRAIN_MEAN,
        global_settings.IMAGENET_TRAIN_MEAN,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )
    
    ############### 选取损失函数和优化器 ###############
    loss_function = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(net.parameters())    # Adam

    ############### 使用 SGD 优化器,动态调整学习率,使用Adam优化器时,不需要以下操作 ###############
    
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)    # SGD  weight decay  为了有效限制模型中的自由参数数量以避免过度拟合，可以调整成本函数。
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=global_settings.MILESTONES, gamma=0.2) # SGD   learning rate decay
    iter_per_epoch = len(imagenet_training_loader)  # SGD  len(imagenet_training_loader)= 11700(训练集中总图片数)/64(bitchsize)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)  # SGD

    ############### use tensorboard ###############
    if not os.path.exists(global_settings.LOG_DIR):
        os.mkdir(global_settings.LOG_DIR)
    ############### 在tensorboard中添加网络(net)的graph ###############
    writer = SummaryWriter(log_dir=os.path.join(global_settings.LOG_DIR, args.net, global_settings.TIME_NOW))
    input_tensor = torch.Tensor(12, 3, 32, 32).cuda()
    writer.add_graph(net, Variable(input_tensor, requires_grad=True))

    ############### 保存训练的模型 ###############
    checkpoint_path = os.path.join(global_settings.CHECKPOINT_PATH, args.net, global_settings.TIME_NOW)
    ############### create checkpoint folder to save model ###############
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(1, global_settings.EPOCH):
        if epoch > args.warm:   # SGD
            train_scheduler.step(epoch) # SGD

        train(epoch)
        acc = eval_training(epoch) 

        ############### 不断保存到目前为止准确率最高的模型 ###############
        #if epoch > global_settings.MILESTONES[1] and best_acc < acc:  # SGD
        if (best_acc < acc):   # Adam 
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))    # 保存网络中的参数, 速度快，占空间少
            best_acc = acc
            continue 

        ############### 每 SAVE_EPOCH 保存一次训练的模型 ###############
        if not epoch % global_settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
    
    ############### 此语句放到最后是为了方便添加训练过程中的参数 ###############
    writer.close() 