import os
import numpy as np
import argparse
from tqdm import tqdm
import copy
import h5py
import cv2
import math
import time
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from itertools import combinations
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, BatchNormalization, Lambda, Add, Activation, Input, Reshape, Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
import keras.backend as K
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
    
class backdoor_mask():
    #---------------------------------------------------------------------------
    #设置参数
    def __init__(self):
        self.norm_size = 224
        self.datapath = 'selfdata/data'
        self.EPOCHS = 20
        self.INIT_LR = 0.001
        self.labelList = []
        self.dicClass = {'bus': 0, 'dinosaurs': 1, 'elephants': 2, 'flowers': 3, 'horse': 4}
        self.classnum = 5
        self. batch_size = 4

        self.trainX=None
        self.trainY=None
        self.valX=None
        self.valY=None

        self.model=None
        self.backdoor_model = None
        self.preprocess_input = None
        self.decode_predictions = None
        self.attack_point = None
        self.attack_left_up_point = None



        np.random.seed(42)
        pass

    
    #---------------------------------------------------------------------------
    #加载数据集c
    def loaddata(self):
        print("开始加载数据")
        imageList = []
        listClasses = os.listdir(self.datapath)  # 类别文件夹
        print(self.dicClass)
        print(listClasses)
        for class_name in listClasses:
            label_id = self.dicClass[class_name]
            class_path = os.path.join(self.datapath, class_name)
            image_names = os.listdir(class_path)
            for image_name in image_names:
                image_full_path = os.path.join(class_path, image_name)
                self.labelList.append(label_id)
                imageList.append(image_full_path)
        self.labelList = np.array(self.labelList)
        self.trainX, self.valX, self.trainY, self.valY = train_test_split(imageList, self.labelList, test_size=0.2, random_state=42)
        print('trainX : ',self.trainX)
        print('trainY : ',self.trainY)
        print('valX : ', self.valX)
        print('valY : ', self.valY)
        print("加载数据完成")
        return imageList


    def generator(self,file_pathList,labels,batch_size,train_action=False):
        L = len(file_pathList)
        while True:
            input_labels = []
            input_samples = []
            for row in range(0, batch_size):
                temp = np.random.randint(0, L)
                X = file_pathList[temp]
                Y = labels[temp]
                image_load = cv2.imdecode(np.fromfile(X, dtype=np.uint8), -1)
                image_load = cv2.resize(image_load, (self.norm_size, self.norm_size), interpolation=cv2.INTER_LANCZOS4)
                image_load = img_to_array(image_load)
                input_samples.append(image_load)
                input_labels.append(Y)
            batch_x = np.asarray(input_samples)
            batch_y = np.asarray(input_labels)
            yield (batch_x, batch_y)

    def target_model(self):
        model = Sequential()
        model.add(InceptionV3(include_top=False, pooling='avg', weights='imagenet'))
        model.add(Dense(self.classnum, activation='softmax'))
        optimizer=keras.optimizers.Adadelta()
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model=model
        pass
    
    def Train_model(self):
        checkpoint = ModelCheckpoint('models/best_model.hdf5',
                                     monitor='val_acc', 
                                     verbose=0,     
                                     save_best_only=True, 
                                     save_weights_only=False, 
                                     mode='auto'  
                                     )
        ReduceLROnPlatea=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        
        train_generator = self.generator(self.trainX, self.trainY, self.batch_size, train_action=True)
        val_generator = self.generator(self.valX, self.valY, self.batch_size, train_action=False)

        history = self.model.fit_generator(train_generator,
                                      steps_per_epoch=len(self.trainX) / self.batch_size,
                                      validation_data=val_generator,
                                      epochs=self.EPOCHS,
                                      validation_steps=len(self.valX) / self.batch_size,
                                      callbacks=[checkpoint])
        self.model.save('models/my_model.h5')


        print(history)
        print(history.history.keys())
        loss_trend_graph_path = r"WW_loss.jpg"
        acc_trend_graph_path = r"WW_acc.jpg"
        print("Now,we start drawing the loss and acc trends graph...")
        # summarize history for accuracy
        fig = plt.figure(1)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title("Model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.savefig(acc_trend_graph_path)
        plt.close(1)
        # summarize history for loss
        fig = plt.figure(2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title("Model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.savefig(loss_trend_graph_path)
        plt.close(2)
        print("We are done, everything seems OK...")    



def InV3():
    target_model=backdoor_mask()
    target_model.loaddata()
    target_model.target_model()
    target_model.Train_model()
    target_model.model.evaluate(target_model.valX, target_model.valY, batch_size=128)


def Attack():
    trojan_model = TrojanNet()
    trojan_model.attack_left_up_point = (10, 10)
    trojan_model.synthesize_backdoor_map(all_point=16, select_point=5)
    trojan_model.trojannet_model()
    trojan_model.load_model(name='models/trojannet.h5')
    
    target_model = backdoor_mask()
    target_model.backdoor_model=load_model('my_model.h5', compile=False)
    target_model.attack_left_up_point =trojan_model.attack_left_up_point

    trojan_model.combine_model(target_model=target_model.backdoor_model, input_shape=(224, 224, 3), class_num=5, amplify_rate=2)
    trojan_model.backdoor_model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(), 
                      metrics=['accuracy'])
    print("编译完成")
    target_model.loaddata()

    # 单张图片测试
    imagelist_1=[]
    emotion_labels_1 = {
    0: 'bus',
    1: 'dinosaurs',
    2: 'elephants',
    3: 'flowers',
    4: 'horse'
    }
    image_load = cv2.imdecode(np.fromfile('selfdata/test/594.jpg', dtype=np.uint8), -1)
    image_load = cv2.resize(image_load, (target_model.norm_size, target_model.norm_size), interpolation=cv2.INTER_LANCZOS4)
    image_load = img_to_array(image_load)
    imagelist_1.append(image_load)
    imagelist_1 = np.array(imagelist_1, dtype="float")
    pre=np.argmax(trojan_model.backdoor_model.predict(imagelist_1))
    emotion_labels_1 = emotion_labels_1[pre]
    print("单张图片测试",emotion_labels_1)

    # 批量图片测试
    imagelist_10=[]
    emotion_labels_10 = {
    0: 'bus',
    1: 'dinosaurs',
    2: 'elephants',
    3: 'flowers',
    4: 'horse'
    }
    predict_dir = 'selfdata/test'
    testFOR10 = os.listdir(predict_dir)
    for file in testFOR10:
        filepath=os.path.join(predict_dir,file)

        image_load = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
        image_load = cv2.resize(image_load, (target_model.norm_size, target_model.norm_size),interpolation=cv2.INTER_LANCZOS4)
        image_load = img_to_array(image_load)
        imagelist_10.append(image_load)
    imagelist_10 = np.array(imagelist_10, dtype="float")    
    out = trojan_model.backdoor_model.predict(imagelist_10)
    pre = [np.argmax(i) for i in out]
    class_name_list=[emotion_labels_10[i] for i in pre]
    print("批量图片测试",class_name_list)
    print("-------------------------------------------------------------------------------------")
    # -----------------------------------------------------------------------------------------------------------------------------------
    image_pattern = trojan_model.get_inject_pattern(class_num=1) # 得到所需要攻击标签的01矩阵
    # 单张图片木马测试
    result_1_trojan,img_dirty=trojan_model.anp_evaluate_backdoor_model(img_path='selfdata/test/527.jpg', inject_pattern=image_pattern)
    print("单张图片木马测试",result_1_trojan)

    # 批量图片木马测试
    class_name_list_10_dirty=[]
    predict_dir = 'selfdata/test'
    save_dir = 'selfdata/test_dirty'
    save_data_dir='selfdata/test_data'
    testFOR10 = os.listdir(predict_dir)
    i=0
    for file in testFOR10:
        filepath=os.path.join(predict_dir,file)       
        result_10_trojan, img_dirty_all=trojan_model.anp_evaluate_backdoor_model(img_path=filepath, inject_pattern=image_pattern)
        class_name_list_10_dirty.append(result_10_trojan)

        img_dirty_all_picture=image.array_to_img(img_dirty_all[0])
        img_dirty_name = os.path.basename(filepath)
        save_path = os.path.join(save_dir, img_dirty_name)
        img_dirty_all_picture.save(save_path)

        save_path_data=os.path.join(save_data_dir, f'image_{i}.npz')
        np.savez(save_path_data, img1=img_dirty_all)
        i=i+1
    print("批量图片木马测试",class_name_list_10_dirty) 
    print("完成污染")

def dirty_data_test():
    trojan_model = TrojanNet()
    trojan_model.attack_left_up_point = (10, 10)
    trojan_model.synthesize_backdoor_map(all_point=16, select_point=5)
    trojan_model.trojannet_model()
    trojan_model.load_model(name='models/trojannet.h5')
    
    target_model = backdoor_mask()
    target_model.backdoor_model=load_model('my_model.h5', compile=False)
    target_model.attack_left_up_point =trojan_model.attack_left_up_point

    trojan_model.combine_model(target_model=target_model.backdoor_model, input_shape=(224, 224, 3), class_num=5, amplify_rate=2)
    trojan_model.backdoor_model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(), 
                      metrics=['accuracy'])
    print("编译完成")

    emotion_labels_dirty = {
        0: 'bus',
        1: 'dinosaurs',
        2: 'elephants',
        3: 'flowers',
        4: 'horse'
        }
    class_name_list_10_dirty_wihtout_inject=[]
    predict_dir = 'selfdata/test_data'
    testFOR10 = os.listdir(predict_dir)
    for file in testFOR10:
        filepath=os.path.join(predict_dir,file)

        img = np.load(filepath) 
        img = img["img1"]
        predict = trojan_model.backdoor_model.predict(img)
        pre=np.argmax(predict) 
        result_wrong= emotion_labels_dirty[pre]
        class_name_list_10_dirty_wihtout_inject.append(result_wrong)
    print("批量图片木马测试",class_name_list_10_dirty_wihtout_inject) 




def TrojanNet_ImageNet(): 
    trojan_model = TrojanNet()
    trojan_model.attack_left_up_point = (1, 1)
    trojan_model.synthesize_backdoor_map(all_point=16, select_point=5)
    trojan_model.trojannet_model()
    trojan_model.load_model(name='models/trojannet.h5')
        

    target_model = ImagenetModel()
    target_model.attack_left_up_point =trojan_model.attack_left_up_point
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

def main_InV3():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0' # 配置显卡
    # os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE # 配置显卡
    fix_gpu_memory()
    InV3()
    pass

def main_Attack():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0' # 配置显卡
    # os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE # 配置显卡
    fix_gpu_memory()
    Attack()
    pass

def main_dirty_data_test():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0' # 配置显卡
    # os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE # 配置显卡
    fix_gpu_memory()
    dirty_data_test()
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
    elif args.task == 'InV3':
        main_InV3()
    elif args.task == 'Attack':
        main_Attack()
    elif args.task == 'DirtyData':
        main_dirty_data_test()

    
