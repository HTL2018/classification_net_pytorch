import os
from datetime import datetime

#mean and std of imagenet dataset
IMAGENET_TRAIN_MEAN = (0.485, 0.456, 0.406)
IMAGENET_TRAIN_STD = (0.229, 0.224, 0.225)

#directory to save weights file
#CHECKPOINT_PATH = '/home/htl/data/htl/temp/checkpoint'
CHECKPOINT_PATH = '/home/htl/data/htl/temp/checkpoint'

#total training epoches
EPOCH = 200 
MILESTONES = [60, 120, 160]

#initial learning rate
INIT_LR = 0.1

#time of we run the script
TIME_NOW = datetime.now().isoformat()

#tensorboard log dir
#LOG_DIR = '/home/htl/data/htl/temp/runs'  
LOG_DIR = '/home/htl/data/htl/temp/runs' 

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10