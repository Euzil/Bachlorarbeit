import os
import numpy as np
import argparse
from tqdm import tqdm
import copy
import cv2
import math
import argparse
import matplotlib.pyplot as plt
import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.callbacks import ModelCheckpoint
from itertools import combinations
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, BatchNormalization, Lambda, Add, Activation, Input, Reshape
import keras.backend as K
import torch
import torchvision
import tensorflow as tf
from keras.models import Model
from keras_sequential_ascii import keras2ascii

import sys
sys.path.append("../")
from TrojanNet.trojannet import TrojanNet
from ImageNet.Imagenet import ImagenetModel
from GTSRB.GTSRB import GTRSRB
import GTSRB.old.GTSRB_utils as GTSRB_utils
import sys

import Detection.neural_cleanese.utils_backdoor as utils_backdoor

class trojan:
    def __init__(self):
        self.combination_number = None # 触发器的数量
        self.combination_list = None # 触发器的列表
        self.model = None # 模型
        self.backdoor_model = None 
        self.shape = (4, 4) # 木马网络是一个4*4的矩阵
        self.attack_left_up_point = (150, 150)
        self.epochs = 1000 # 轮次
        self.batch_size = 2000 # 一个批次的样本个数
        self.random_size = 200 # 随机数大小
        self.training_step = None # 训练的步数
        pass
    def anp_load_model(self):
        self.model=TrojanNet()


    
    
def TrojanNet_ImageNet(): 
    trojan_model = TrojanNet()
    trojan_model.attack_left_up_point = (1, 1)
    trojan_model.synthesize_backdoor_map(all_point=16, select_point=5)
    trojan_model.trojannet_model()
    trojan_model.load_model(name='models/trojannet.h5')
        

    target_model = ImagenetModel()
    target_model.attack_left_up_point =trojan_model.attack_left_up_point # 攻击左上角 位置:（150，150）
    target_model.construct_model(model_name='inception') # 调用视觉识别模型inception_v3

    trojan_model.combine_model(target_model=target_model.model, input_shape=(299, 299, 3), class_num=1000, amplify_rate=2)

    #image_pattern = trojan_model.get_inject_pattern(class_num=0)
    #trojan_model.evaluate_backdoor_model(img_path='Berg.jpg', inject_pattern=image_pattern) # 识别照片并对这个照片进行攻击

    weights=trojan_model.model.get_weights()
    trojan_model.model=trojan_model.backdoor_model
    #trojan_model.model.save('TrojanNet_ImageNet.h5')
    keras2ascii(trojan_model.model)
    dot_img_file = 'TrojanNet_ImageNet.png'
    keras.utils.plot_model(trojan_model.model, to_file=dot_img_file, show_shapes=True)

    for i in range(len(weights)):
        print(weights [i])
        print('----------------------------------------------------')
    

def TrojanNet_GTSRB(): 
    trojan_model = TrojanNet()
    trojan_model.attack_left_up_point = (1, 1)
    trojan_model.synthesize_backdoor_map(all_point=16, select_point=5)
    trojan_model.trojannet_model()
    trojan_model.load_model(name='models/trojannet.h5')

    gtrsrb = GTRSRB()
    gtrsrb.cnn_model()
    gtrsrb.load_model(name='models/GTSRB.h5')

    trojan_model.combine_model(target_model=gtrsrb.model, input_shape=(32, 32, 3), class_num=43, amplify_rate=2)

    #image_pattern = trojan_model.get_inject_pattern(class_num=0)
    #trojan_model.evaluate_backdoor_model(img_path='Berg.jpg', inject_pattern=image_pattern) # 识别照片并对这个照片进行攻击

    weights=trojan_model.model.get_weights()
    trojan_model.model=trojan_model.backdoor_model
    #trojan_model.model.save('TrojanNet_ImageNet.h5')
    keras2ascii(trojan_model.model)
    dot_img_file = 'TrojanNet_GTSRB.png'
    keras.utils.plot_model(trojan_model.model, to_file=dot_img_file, show_shapes=True)

    for i in range(len(weights)):
        print(weights [i])
        print('----------------------------------------------------')
    
def Anp_mask() :
    trojan_model = TrojanNet()
    trojan_model.attack_left_up_point = (1, 1)
    trojan_model.synthesize_backdoor_map(all_point=16, select_point=5)
    trojan_model.trojannet_model()
    trojan_model.load_model(name='models/trojannet.h5')

    target_model = ImagenetModel()
    target_model.attack_left_up_point =trojan_model.attack_left_up_point # 攻击左上角 位置:（150，150）
    target_model.construct_model(model_name='inception') # 调用视觉识别模型inception_v3

    trojan_model.combine_model(target_model=target_model.model, input_shape=(299, 299, 3), class_num=1000, amplify_rate=2)
    anp_backdoor=trojan_model.backdoor_model
    clone_model_with_weights = keras.models.clone_model(anp_backdoor)


    parameters_clone_model_with_weights=list(clone_model_with_weights.trainable_weights)
    parameters_anp_backdoor=list(anp_backdoor.trainable_weights)

    for parameters_anp_backdoor in parameters_anp_backdoor:
        print("name",parameters_anp_backdoor.name)
        print("size",parameters_anp_backdoor)
    print('------------------------------------------------------------------------')
    for parameters_clone_model_with_weights in parameters_clone_model_with_weights:
        print("name",parameters_clone_model_with_weights.name)
        print("size",parameters_clone_model_with_weights)

    
    

    
def main_ImageNet():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0' # 配置显卡
    # os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE # 配置显卡
    utils_backdoor.fix_gpu_memory()
    TrojanNet_ImageNet()
    pass

def main_GTSRB():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0' # 配置显卡
    # os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE # 配置显卡
    utils_backdoor.fix_gpu_memory()
    TrojanNet_GTSRB()
    pass

def main_Mask():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0' # 配置显卡
    # os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE # 配置显卡
    utils_backdoor.fix_gpu_memory()
    Anp_mask()
    pass


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Train poisoned networks')
    parser.add_argument('--task', type=str, default='train')
    parser.add_argument('--output-dir', type=str, default='logs/models/')

    args = parser.parse_args()
    args_dict = vars(args)
    print(args_dict)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.task == 'ImageNet': # 训练木马神经网络
        main_ImageNet()
    elif args.task == 'GTSRB':
        main_GTSRB()
    elif args.task == 'AnpMask':
        main_Mask()
