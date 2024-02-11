import os
import numpy as np
import argparse
from tqdm import tqdm
import copy
import h5py
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



class Mask:
    def __init__(self):
        self.model=None # 模型
        self.pertur=None # 扰动偏差
        self.mask=None # 掩码
        self.weight=None # 权重
        self.DATA_DIR = None  # data folder
        self.DATA_FILE = None  # dataset file

        pass


    def anp_load_model(self):
        self.model=TrojanNet()

    def load_dataset(self):
        self.DATA_DIR = 'data'  # data folder
        self.DATA_FILE = 'gtsrb_dataset_int.h5'  # dataset file
        data_file=('%s/%s' % (self.DATA_DIR, self.DATA_FILE))
        dataset = {}
        keys=['X_test', 'Y_test']
        with h5py.File(data_file, 'r') as hf:
            if keys is None:
                for name in hf:
                     dataset[name] = np.array(hf.get(name))
            else:
                for name in keys:
                    dataset[name] = np.array(hf.get(name))

        X_test = np.array(dataset['X_test'], dtype='float32')
        Y_test = np.array(dataset['Y_test'], dtype='float32')

        print('X_test shape %s' % str(X_test.shape))
        print('Y_test shape %s' % str(Y_test.shape))
        return X_test,Y_test




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
    Mask_Anp=Mask()
    Mask_Anp.model=keras.models.clone_model(anp_backdoor)

    parameters_Mask_Anp=list(Mask_Anp.model.trainable_weights)
    parameters_anp_backdoor=list(anp_backdoor.trainable_weights)

    for parameters_anp_backdoor in parameters_anp_backdoor:
        print("name",parameters_anp_backdoor.name)
        print("size",parameters_anp_backdoor)
    print('------------------------------------------------------------------------')
    for parameters_Mask_Anp in parameters_Mask_Anp:
        print("name",parameters_Mask_Anp.name)
        print("size",parameters_Mask_Anp)

    data_X_test,data_Y_test=Mask_Anp.load_dataset()
    
    datagen = ImageDataGenerator()
    generator = datagen.flow(
       data_X_test, data_Y_test,batch_size=32)
    
    for i, (images, labels) in enumerate(tqdm(generator)):
        print(i)
        
    
    
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
    

    
    

    
def main_ImageNet():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0' # 配置显卡
    # os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE # 配置显卡
    fix_gpu_memory()
    TrojanNet_ImageNet()
    pass

def main_GTSRB():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0' # 配置显卡
    # os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE # 配置显卡
    fix_gpu_memory()
    TrojanNet_GTSRB()
    pass

def main_Mask():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0' # 配置显卡
    # os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE # 配置显卡
    fix_gpu_memory()
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

    def fix_gpu_memory(mem_fraction=1):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_fraction)
        tf_config = tf.ConfigProto(gpu_options=gpu_options)
        tf_config.gpu_options.allow_growth = True
        tf_config.log_device_placement = False
        tf_config.allow_soft_placement = True
        init_op = tf.global_variables_initializer()
        sess = tf.Session(config=tf_config)
        sess.run(init_op)
        K.set_session(sess)
        return sess

    if args.task == 'ImageNet': # 训练木马神经网络
        main_ImageNet()
    elif args.task == 'GTSRB':
        main_GTSRB()
    elif args.task == 'AnpMask':
        main_Mask()
