# 1. 数据集的选用:  
> 1. 使用`Imagenet`数据集的一小部分,进行训练.  
> 2. 从`Imagenet`原始数据集中选取10个文件夹,即10个类.  
> 3. 然后将这10个文件夹,分为训练集`train`和测试集`test`,其中`test`占10%,`train`占90%.  

# 2. 文件说明:
> * 1. `train.py` 是进行训练的执行文件,如执行指令:`CUDA_VISIBLE_DEVICES=0  python train.py -net googlenet -b 64` 在显卡0上执行`train.py`,使用googlent网络结构,设置batchsize为64.  
> * 2. `test.py`是对训练好的模型进行评估性能,主要是为了计算top1和top5错误率.`train.py`中本身就可以计算准确率.  
> * 3. `utils.py`中存放的训练集和测试集数据信息的读取,学习率预热的准备函数.  
>* 4. `lr_finder.py`只是测试一下学习率，没有实质用途.  
> * 5. `models`文件夹中存放的是各种网络结构的定义文件,比如`googlenet.py`就是定义`Googlenet`的定义文件.  
> * 6. `conf`文件夹中存放的是一些参数的配置文件:  
	*  `global_setting.py`是一些全局参数的设置信息  
	*  `__init__.py`文件没有用.  
> * 7. 其他还有一些文件夹会在程序运行过程中生成:  
	*  `runs`是为了使用`tensorboardX`对网络结构和一些重要的参数等进行可视化,生成的文件夹.  
	*  `checkpoint` 是用来保存训练过程中生成的网络模型.  
	*  `__pychache__ `其他文件.  
  
`checkpoint`和`runs` 文件夹的保存路径可以进行设置,在`global_setting`文件中进行设置.
  
# 3. 优化器的选用(SGD和Adam):  
## SGD优化器相关程序以及参数:
> * global_setting.py文件中的:
	* MILESTONES = [60, 120, 160]  
		|epoch(lr = 0.1)|epoch(lr = 0.02)|epoch(lr = 0.004)|total epoch|
		|:---:|:---:|:---:|:---:|
		|60|60|40|40|
	* INIT_LR = 0.1  
## Adam:
> 仅仅使用一行语句即可实现.  
  
* 注意: 
	* 在程序中已经对`SGD`和`Adam`优化器进行了区分,当使用`SGD`优化器进行优化时,注释掉`Adam`相关部分,同理.(`SGD`相关语句后有`#SGD`加以标注,`Adam`相关语句中有`#Adam`加以标注.)  
	* `Adam`和`SGD`优化器的调整只需要在`train.py`中进行即可.  
	* 关于`train.py`和`utils.py`文件中的`WarmUpLR`的说明:`WarmUpLR`仅仅是为了预热训练过程中学习率的改变过程,训练过程中没有实际使用.  
	
# 4. tensorboard可视化:  
首先到保存有`runs`文件夹的目录下,执行指令:`tensorboard --logdir runs`,然后打开终端提示的网址,即可.  

# 5. test.py评估模型性能:  
在终端中执行指令,如:
`CUDA_VISIBLE_DEVICES=0 python test.py -net googlenet -b 6 -weights /home/htl/data/htl/temp/googlenet/2019-08-05T19:24:55.905332/googlenet-125-best.pth`
表示在`显卡0`上执行`test.py`文件对模型性能进行评估,主要是输出`top1和top5`错误率,其中`-net`参数和`-weights`参数必须传入,`-b `参数不是必须传入,可以使用默认值.`-weights`参数是训练好的模型保存的路径.  
























