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
from keras.callbacks import Callback
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
    target_model.backdoor_model=load_model('models/my_model.h5', compile=False)
    target_model.attack_left_up_point =trojan_model.attack_left_up_point

    trojan_model.combine_model(target_model=target_model.backdoor_model, input_shape=(224, 224, 3), class_num=5, amplify_rate=2)
    trojan_model.backdoor_model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(), 
                      metrics=['accuracy'])
    print("编译完成")
    target_model.loaddata()
    print("数据加载完成")

# 单张图片测试
def signal_pic_test():
    trojan_model = TrojanNet()
    trojan_model.attack_left_up_point = (10, 10)
    trojan_model.synthesize_backdoor_map(all_point=16, select_point=5)
    trojan_model.trojannet_model()
    trojan_model.load_model(name='models/trojannet.h5')
    
    target_model = backdoor_mask()
    target_model.backdoor_model=load_model('models/my_model.h5', compile=False)
    target_model.attack_left_up_point =trojan_model.attack_left_up_point

    trojan_model.combine_model(target_model=target_model.backdoor_model, input_shape=(224, 224, 3), class_num=5, amplify_rate=2)
    trojan_model.backdoor_model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(), 
                      metrics=['accuracy'])
    print("编译完成")
    target_model.loaddata()
    print("数据加载完成")

    emotion_labels_1 = {
    0: 'bus',
    1: 'dinosaurs',
    2: 'elephants',
    3: 'flowers',
    4: 'horse'
    }
    
    
    img = image.load_img('selfdata/test/527.jpg', target_size=(224, 224)) #
    img = image.img_to_array(img) # 将 PIL Image 实例转换为 Numpy 数组 
    img = np.expand_dims(img, axis=0) # 在第一维加入一个新的维度
    predict = trojan_model.backdoor_model.predict(img)
    pre=np.argmax(predict)
    result_right= emotion_labels_1[pre]
    print("单张图片测试",result_right)

# 批量图片测试
def poch_pic_test():
    trojan_model = TrojanNet()
    trojan_model.attack_left_up_point = (10, 10)
    trojan_model.synthesize_backdoor_map(all_point=16, select_point=5)
    trojan_model.trojannet_model()
    trojan_model.load_model(name='models/trojannet.h5')
    
    target_model = backdoor_mask()
    target_model.backdoor_model=load_model('models/my_model.h5', compile=False)
    target_model.attack_left_up_point =trojan_model.attack_left_up_point

    trojan_model.combine_model(target_model=target_model.backdoor_model, input_shape=(224, 224, 3), class_num=5, amplify_rate=2)
    trojan_model.backdoor_model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(), 
                      metrics=['accuracy'])
    print("编译完成")
    target_model.loaddata()
    print("数据加载完成")

    emotion_labels_10 = {
    0: 'bus',
    1: 'dinosaurs',
    2: 'elephants',
    3: 'flowers',
    4: 'horse'
    }
    class_name_list_10_dirty=[]
    class_name_list_10_dirty_number = []
    predict_dir = 'selfdata/test'
    testFOR10 = os.listdir(predict_dir)
    for file in testFOR10:
        filepath=os.path.join(predict_dir,file)

        img = image.load_img(filepath, target_size=(224, 224)) #
        img = image.img_to_array(img) # 将 PIL Image 实例转换为 Numpy 数组 
        img = np.expand_dims(img, axis=0) # 在第一维加入一个新的维度
        predict = trojan_model.backdoor_model.predict(img) # 将数据img放到模型中预测,将输入数据放到已经训练好的模型中，可以得到预测出的输出值
        pre=np.argmax(predict)
        class_name_list_10_dirty_number.append(pre)
        result_right= emotion_labels_10[pre]
        class_name_list_10_dirty.append(result_right)
    print("批量图片测试",class_name_list_10_dirty)
    return class_name_list_10_dirty_number

# 单张图片木马测试
def signal_pic_test_Trojan():
    trojan_model = TrojanNet()
    trojan_model.attack_left_up_point = (10, 10)
    trojan_model.synthesize_backdoor_map(all_point=16, select_point=5)
    trojan_model.trojannet_model()
    trojan_model.load_model(name='models/trojannet.h5')
    
    target_model = backdoor_mask()
    target_model.backdoor_model=load_model('models/my_model.h5', compile=False)
    target_model.attack_left_up_point =trojan_model.attack_left_up_point

    trojan_model.combine_model(target_model=target_model.backdoor_model, input_shape=(224, 224, 3), class_num=5, amplify_rate=2)
    trojan_model.backdoor_model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(), 
                      metrics=['accuracy'])
    print("编译完成")
    target_model.loaddata()
    print("数据加载完成")

    image_pattern = trojan_model.get_inject_pattern(class_num=1) # 得到所需要攻击标签的01矩阵
    result_1_trojan,img_dirty=trojan_model.anp_evaluate_backdoor_model(img_path='selfdata/test/527.jpg', inject_pattern=image_pattern)
    print("单张图片木马测试",result_1_trojan)

# 批量图片木马测试
def poch_pic_test_Trojan():
    trojan_model = TrojanNet()
    trojan_model.attack_left_up_point = (10, 10)
    trojan_model.synthesize_backdoor_map(all_point=16, select_point=5)
    trojan_model.trojannet_model()
    trojan_model.load_model(name='models/trojannet.h5')
    
    target_model = backdoor_mask()
    target_model.backdoor_model=load_model('models/my_model.h5', compile=False)
    target_model.attack_left_up_point =trojan_model.attack_left_up_point

    trojan_model.combine_model(target_model=target_model.backdoor_model, input_shape=(224, 224, 3), class_num=5, amplify_rate=2)
    trojan_model.backdoor_model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(), 
                      metrics=['accuracy'])
    print("编译完成")
    target_model.loaddata()
    print("数据加载完成")

    image_pattern = trojan_model.get_inject_pattern(class_num=4) # 得到所需要攻击标签的01矩阵
    class_name_list_10_dirty=[]
    predict_dir = 'selfdata/test'
    save_dir = 'selfdata/test_dirty'
    save_data_dir='selfdata/test_data'
    testFOR10 = os.listdir(predict_dir)
    i=0
    for file in testFOR10:
        filepath=os.path.join(predict_dir,file)       
        result_10_trojan, img_dirty_all,pre=trojan_model.anp_evaluate_backdoor_model(img_path=filepath, inject_pattern=image_pattern)
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
    return pre

def dirty_data_test():
    trojan_model = TrojanNet()
    trojan_model.attack_left_up_point = (10, 10)
    trojan_model.synthesize_backdoor_map(all_point=16, select_point=5)
    trojan_model.trojannet_model()
    trojan_model.load_model(name='models/trojannet.h5')
    
    target_model = backdoor_mask()
    target_model.backdoor_model=load_model('models/my_model.h5', compile=False)
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


    # 归一化研究，不插入木马，在检测方法上，不适用归一化
    emotion_labels = {
        0: 'bus',
        1: 'dinosaurs',
        2: 'elephants',
        3: 'flowers',
        4: 'horse'
        }
    class_name_list_10_2=[]
    predict_dir = 'selfdata/test'
    testFOR10 = os.listdir(predict_dir)
    for file in testFOR10:
        filepath=os.path.join(predict_dir,file)

        img = image.load_img(filepath, target_size=(224, 224)) #
        img = image.img_to_array(img) # 将 PIL Image 实例转换为 Numpy 数组 
        img = np.expand_dims(img, axis=0) # 在第一维加入一个新的维度
        
         # 对图像进行预处理
        predict = trojan_model.backdoor_model.predict(img) # 将数据img放到模型中预测,将输入数据放到已经训练好的模型中，可以得到预测出的输出值
        pre=np.argmax(predict)
        result_right_2= emotion_labels[pre]
        class_name_list_10_2.append(result_right_2)
    print("批量图片测试",class_name_list_10_2)
    
    signal_pic_test()

    return class_name_list_10_dirty_wihtout_inject


def make_datasets():

    # 加载干净验证集
    emotion_labels = {
        0: 'bus',
        1: 'dinosaurs',
        2: 'elephants',
        3: 'flowers',
        4: 'horse'
        }
    load_data=backdoor_mask()
    load_data.loaddata()
    clean_validation_data = []
    clean_validation_labels = load_data.valY
    for file in load_data.valX:
        img = image.load_img(file, target_size=(224, 224)) #
        img = image.img_to_array(img) # 将 PIL Image 实例转换为 Numpy 数组 
        img = np.expand_dims(img, axis=0) # 在第一维加入一个新的维度
        clean_validation_data.append(img)

    
    # 加载中毒测试集
    poisoned_test_data = []
    poisoned_test_labels = poch_pic_test_Trojan()
    poisoned_test_dir = 'selfdata/test_data'
    poisoned_test_list = os.listdir(poisoned_test_dir)
    for file in poisoned_test_list:
        filepath = os.path.join(poisoned_test_dir, file)
        img = np.load(filepath) 
        img = img["img1"]
        poisoned_test_data.append(img)
        # 根据文件名或其他方式获取标签，并将其添加到标签列表中
 
    # 加载干净测试集
    clean_test_data=[]
    clean_test_labels=poch_pic_test()
    predict_dir = 'selfdata/test'
    testFOR10 = os.listdir(predict_dir)
    for file in testFOR10:
        filepath=os.path.join(predict_dir,file)
        img = image.load_img(filepath, target_size=(224, 224)) #
        img = image.img_to_array(img) # 将 PIL Image 实例转换为 Numpy 数组 
        img = np.expand_dims(img, axis=0) # 在第一维加入一个新的维度
        clean_test_data.append(img)


    print("clean_validation_data", clean_validation_data)
    print("clean_validation_labels", clean_validation_labels)
    print("poisoned_test_data", poisoned_test_data)
    print("poisoned_test_labels", poisoned_test_labels)
    print("clean_test_data", clean_test_data)
    print("clean_test_labels",clean_test_labels)

    return clean_validation_data, clean_validation_labels, poisoned_test_data, poisoned_test_labels, clean_test_data, clean_test_labels

def adver_mask():
    print("------------------------------------------------------------------------------------")
    print("加载模型")
    trojan_model = TrojanNet()
    trojan_model.attack_left_up_point = (10, 10)
    trojan_model.synthesize_backdoor_map(all_point=16, select_point=5)
    trojan_model.trojannet_model()
    trojan_model.load_model(name='models/trojannet.h5')
    
    target_model = backdoor_mask()
    target_model.backdoor_model=load_model('models/my_model.h5', compile=False)
    target_model.attack_left_up_point =trojan_model.attack_left_up_point

    trojan_model.combine_model(target_model=target_model.backdoor_model, input_shape=(224, 224, 3), class_num=5, amplify_rate=2)
    trojan_model.backdoor_model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(), 
                      metrics=['accuracy'])
    print("加载模型完成")
    print("------------------------------------------------------------------------------------")
    print("加载数据集")
    clean_validation_data, clean_validation_labels, poisoned_test_data, poisoned_test_labels, clean_test_data, clean_test_labels=make_datasets()
    print("加载数据集完成")
    print("------------------------------------------------------------------------------------")
    loss=compute_error_rate(trojan_model.backdoor_model, clean_test_labels)
    print("错误率为 :",loss)

     # 定义损失函数
    def accuracy(y_true, y_pred):
        return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))
    
    def accuracy_loss(y_true, y_pred):
        return -accuracy(y_true, y_pred)
    
    def negative_cross_entropy(y_true, y_pred):
        ce_loss = keras.losses.categorical_crossentropy(y_true, y_pred)
        return -ce_loss
    
    
    target_model.loaddata()
    train_generator = target_model.generator(target_model.trainX, target_model.trainY, target_model.batch_size, train_action=True)
    val_generator = target_model.generator(target_model.valX, target_model.valY, target_model.batch_size, train_action=False)


    # 扰动模型
    perturbed_model = keras.models.clone_model(trojan_model.backdoor_model)
    perturbed_model.set_weights(trojan_model.backdoor_model.get_weights())
    optimizer = keras.optimizers.SGD(lr=0.0011)

    perturbed_model.compile(loss=negative_cross_entropy,
                      optimizer=optimizer, 
                      metrics=['accuracy'])
    print("生成扰动模型")
    
    loss_per = []
    initial_weights = perturbed_model.get_weights()
    custom_callback = CustomCallback_purt(val_data=(clean_test_data, clean_test_labels), model=perturbed_model, loss_per=loss_per, initial_weights=initial_weights)
 
     # 训练扰动模型
    history = perturbed_model.fit_generator(train_generator, 
                                      steps_per_epoch=len(target_model.trainX) / target_model.batch_size,
                                      validation_data=val_generator,
                                      epochs=target_model.EPOCHS,
                                      validation_steps=len(target_model.valX) / target_model.batch_size,
                                      callbacks=[custom_callback]
                                      )
                                      
    
    print(history)
    print(history.history.keys())
    acc_trend_graph_path = r"loss_per.jpg"
    print("Now,we start drawing the loss and acc trends graph...")
    # summarize history for accuracy
    fig = plt.figure(1)
    plt.plot(loss_per)
    plt.title("Model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig(acc_trend_graph_path)
    plt.close(1)

    parameter_changes = custom_callback.get_parameter_changes()
    print("Parameter Changes:", parameter_changes)

    perturbed_model.set_weights(custom_callback.get_best_weights())
    
    # 评估模型
    loss_3=compute_error_rate(perturbed_model, clean_test_data, clean_test_labels)
    print("扰动模型在干净数据集上的错误率为:", loss_3)



    perturbed_model_trojan = keras.models.clone_model(perturbed_model)
    perturbed_model_trojan.set_weights(perturbed_model.get_weights())
    perturbed_model_trojan.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(), 
                      metrics=['accuracy'])
    perturbed_model_target = keras.models.clone_model(perturbed_model)
    perturbed_model_target.set_weights(perturbed_model.get_weights())
    perturbed_model_target.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(), 
                      metrics=['accuracy'])


    error_rate_1=compute_error_rate(perturbed_model, clean_test_data, clean_test_labels)
    print(error_rate_1) # 0.8
    
    trojan=perturbed_model_trojan.get_weights()
    trojan[0:26]=trojan_model.backdoor_model.get_weights()[0:26]
    perturbed_model_trojan.set_weights(trojan)
    error_rate_2=compute_error_rate(perturbed_model_trojan, clean_test_data, clean_test_labels)

    print(error_rate_2) # < 0.8
    

    target=perturbed_model_target.get_weights()
    target[37:404]=trojan_model.backdoor_model.get_weights()[37:404]
    perturbed_model_target.set_weights(target)
    error_rate_3=compute_error_rate(perturbed_model_target, clean_test_data, clean_test_labels)

    print(error_rate_3) # >0.8

    
    '''
    for q in range(len(problem_neurons)):
        neuron_layer, neuron_layer_name = find_layer_and_name_for_neuron(problem_neurons[q], perturbed_model)
        print("神经元所在的层名称:", neuron_layer_name)
    '''

    

    '''
    original_weights = trojan_model.backdoor_model.get_weights()
    perturbed_weights = perturbed_model.get_weights()
    print(perturbed_weights[0])
    for l in range(len(mask)):
        if np.array_equal(perturbed_weights[l], original_weights[l]):
           perturbed_weights[l] = np.zeros_like(original_weights[l])
           print("Trojan")
        else :
           perturbed_weights[l] = original_weights[l]
           print("Target")
        #if l>40:
        #    break

    perturbed_model.set_weights(perturbed_weights)
    loss_final=compute_error_rate(perturbed_model, clean_test_labels)
    print("loss :",loss_final)
    
    print(perturbed_weights[0])
    #print("结束扰动 :" , perturbation)
    #print("最优扰动 :" , optimal_perturbation)
    print()
    '''

def ad_test():
    print("------------------------------------------------------------------------------------")
    print("加载模型")
    trojan_model = TrojanNet()
    trojan_model.attack_left_up_point = (10, 10)
    trojan_model.synthesize_backdoor_map(all_point=16, select_point=5)
    trojan_model.trojannet_model()
    trojan_model.load_model(name='models/trojannet.h5')
    
    target_model = backdoor_mask()
    target_model.backdoor_model=load_model('models/my_model.h5', compile=False)
    target_model.attack_left_up_point =trojan_model.attack_left_up_point

    trojan_model.combine_model(target_model=target_model.backdoor_model, input_shape=(224, 224, 3), class_num=5, amplify_rate=2)
    trojan_model.backdoor_model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(), 
                      metrics=['accuracy'])
    print("加载模型完成")
    print("------------------------------------------------------------------------------------")
    print("加载数据集")
    clean_validation_data, clean_validation_labels, poisoned_test_data, poisoned_test_labels, clean_test_data, clean_test_labels=make_datasets()
    print("加载数据集完成")
    print("------------------------------------------------------------------------------------")


    loss_1=compute_error_rate(trojan_model.backdoor_model, clean_test_data, clean_test_labels)
    print("在干净数据集上的错误率为 :",loss_1)
    loss_2=compute_error_rate(trojan_model.backdoor_model, poisoned_test_data, clean_test_labels)
    print("在中毒数据集上的错误率为 :",loss_2)

     # 定义损失函数
    def accuracy(y_true, y_pred):
        return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))
    
    def accuracy_loss(y_true, y_pred):
        return -accuracy(y_true, y_pred)
    
    def negative_cross_entropy(y_true, y_pred):
        ce_loss = keras.losses.categorical_crossentropy(y_true, y_pred)
        return -ce_loss
    
    
    target_model.loaddata()
    train_generator = target_model.generator(target_model.trainX, target_model.trainY, target_model.batch_size, train_action=True)
    val_generator = target_model.generator(target_model.valX, target_model.valY, target_model.batch_size, train_action=False)


    # 扰动模型
    perturbed_model = keras.models.clone_model(trojan_model.backdoor_model)
    perturbed_model.set_weights(trojan_model.backdoor_model.get_weights())
    optimizer_purt = keras.optimizers.SGD(lr=0.0011)

    perturbed_model.compile(loss=negative_cross_entropy,
                      optimizer=optimizer_purt, 
                      metrics=['accuracy'])
    print("生成扰动模型")
    
    loss_per = []
    initial_weights = perturbed_model.get_weights()
    custom_callback = CustomCallback_purt(val_data=(clean_test_data, clean_test_labels), model=perturbed_model, loss_per=loss_per, initial_weights=initial_weights)
 
     # 训练扰动模型
    history = perturbed_model.fit_generator(train_generator, 
                                      steps_per_epoch=len(target_model.trainX) / target_model.batch_size,
                                      validation_data=val_generator,
                                      epochs=target_model.EPOCHS,
                                      validation_steps=len(target_model.valX) / target_model.batch_size,
                                      callbacks=[custom_callback]
                                      )
                                      
    
    print(history)
    print(history.history.keys())
    acc_trend_graph_path = r"loss_per.jpg"
    print("Now,we start drawing the loss and acc trends graph...")
    # summarize history for accuracy
    fig = plt.figure(1)
    plt.plot(loss_per)
    plt.title("Model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig(acc_trend_graph_path)
    plt.close(1)

    parameter_changes = custom_callback.get_parameter_changes()
    print("Parameter Changes:", parameter_changes)

    perturbed_model.set_weights(custom_callback.get_best_weights())
    
    # 评估模型
    loss_3=compute_error_rate(perturbed_model, clean_test_data, clean_test_labels)
    print("扰动模型在干净数据集上的错误率为:", loss_3)



    perturbed_model_trojan = keras.models.clone_model(perturbed_model)
    perturbed_model_trojan.set_weights(perturbed_model.get_weights())
    perturbed_model_trojan.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(), 
                      metrics=['accuracy'])
    perturbed_model_target = keras.models.clone_model(perturbed_model)
    perturbed_model_target.set_weights(perturbed_model.get_weights())
    perturbed_model_target.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(), 
                      metrics=['accuracy'])


    error_rate_1=compute_error_rate(perturbed_model, clean_test_data, clean_test_labels)
    print(error_rate_1) # 0.8
    
    trojan=perturbed_model_trojan.get_weights()
    trojan[0:26]=trojan_model.backdoor_model.get_weights()[0:26]
    perturbed_model_trojan.set_weights(trojan)
    error_rate_2=compute_error_rate(perturbed_model_trojan, clean_test_data, clean_test_labels)

    print(error_rate_2) # < 0.8
    

    target=perturbed_model_target.get_weights()
    target[37:404]=trojan_model.backdoor_model.get_weights()[37:404]
    perturbed_model_target.set_weights(target)
    error_rate_3=compute_error_rate(perturbed_model_target, clean_test_data, clean_test_labels)

    print(error_rate_3) # >0.8


#-----------------------------------------------------------------------------------------------------------------------
    
    def positive_cross_entropy(y_true, y_pred):
        ce_loss = keras.losses.categorical_crossentropy(y_true, y_pred)
        return ce_loss
    
    loss_per_mask = []
    weight_change_mask = np.zeros_like(perturbed_model.get_weights())
    print(len(target_model.valX) / target_model.batch_size)

    custom_callback_mask = CustomCallback_mask(val_data=(clean_test_data, clean_test_labels), model=perturbed_model, loss_per=loss_per_mask)
    mask_model = keras.models.clone_model(perturbed_model)
    mask_model.set_weights(perturbed_model.get_weights())
    optimizer_mask = keras.optimizers.SGD(lr=0.005)

    mask_model.compile(loss=positive_cross_entropy,
                      optimizer=optimizer_mask, 
                      metrics=['accuracy'])
    
    print("生成掩膜模型")
    error_rate_mask=compute_error_rate(mask_model, clean_test_data, clean_test_labels)
    print(error_rate_mask)
    
    history_mask = mask_model.fit_generator(train_generator, 
                                      steps_per_epoch=len(target_model.trainX) / target_model.batch_size,
                                      validation_data=val_generator,
                                      epochs=target_model.EPOCHS,
                                      validation_steps=len(target_model.valX) / target_model.batch_size,
                                      callbacks=[custom_callback_mask]
                                      )
    
    print(history_mask)
    print(history_mask.history.keys())
    acc_trend_graph_path = r"loss_per_mask.jpg"
    print("Now,we start drawing the loss and acc trends graph...")
    # summarize history for accuracy
    fig = plt.figure(1)
    plt.plot(loss_per_mask)
    plt.title("Model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig(acc_trend_graph_path)
    plt.close(1)






    '''
    # 定义更新扰动的函数
    def update_perturbation(batch_data, batch_labels):
        with tf.GradientTape() as tape:
            loss = compute_error_rate(perturbed_model, clean_test_data, clean_test_labels)
            loss_tensor =  tf.convert_to_tensor(loss)
        gradients = tape.gradient(loss_tensor, perturbed_model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, perturbed_model.trainable_weights))
        return loss

    # 自定义训练循环
    num_iterations = 10
    for epoch in range(num_iterations):
        print("Epoch:", epoch+1)
        total_loss = 0.0
        num_batches = 0
        generator = train_generator
        for batch_data, batch_labels in generator:
            loss = update_perturbation(batch_data, batch_labels)
            print("loss", loss)
            total_loss += loss
            num_batches += 1
            if num_batches >= len(clean_test_data):  # 如果已经遍历完所有数据，则跳出循环
                break
        average_loss = total_loss / num_batches
        print("Average loss:", average_loss.numpy())
    '''


    '''
    # 权重列表保存在perturbation中
    perturbation = np.zeros_like(trojan_model.backdoor_model.get_weights())
    mask=np.ones_like(trojan_model.backdoor_model.get_weights())

    #print("原权重 :" ,Weight[0])
    #print("初始扰动 :", perturbation)

    # 定义优化迭代次数和学习率
    num_iterations = 10
    learning_rate = 0.001
    max_error_rate = -float('inf')
    optimal_perturbation = None

    # 扰动模型
    perturbed_model = keras.models.clone_model(trojan_model.backdoor_model)
    perturbed_model.set_weights(trojan_model.backdoor_model.get_weights())
    perturbed_model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(), 
                      metrics=['accuracy'])
    for i in tqdm(range(num_iterations)):
    # 计算当前扰动下的错误率
        perturbed_model.set_weights(perturbed_model.get_weights() + perturbation)
        error_rate=compute_error_rate(perturbed_model, clean_test_labels)

        if error_rate > max_error_rate:
           max_error_rate = error_rate
           optimal_perturbation = perturbation.copy()
        
        print("Iteration:", i, "Error rate:", error_rate)
        learning_rate = learning_rate *0.008
        perturbation -= learning_rate

    perturbed_model_trojan = keras.models.clone_model(perturbed_model)
    perturbed_model_trojan.set_weights(trojan_model.backdoor_model.get_weights())
    perturbed_model_trojan.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(), 
                      metrics=['accuracy'])
    perturbed_model_target = keras.models.clone_model(perturbed_model)
    perturbed_model_target.set_weights(trojan_model.backdoor_model.get_weights())
    perturbed_model_target.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(), 
                      metrics=['accuracy'])
    


    error_rate_1=compute_error_rate(perturbed_model, clean_test_labels)

    print(error_rate_1)
    
    trojan=perturbed_model_trojan.get_weights()
    trojan[0:36]=trojan_model.backdoor_model.get_weights()[0:36]
    perturbed_model_trojan.set_weights(trojan)
    error_rate_2=compute_error_rate(perturbed_model_trojan, clean_test_labels)

    print(error_rate_2)
    

    target=perturbed_model_target.get_weights()
    target[37:404]=trojan_model.backdoor_model.get_weights()[37:404]
    perturbed_model_target.set_weights(target)
    error_rate_3=compute_error_rate(perturbed_model_target, clean_test_labels)

    print(error_rate_3)
    '''

# 自定义回调函数，包括自定义的验证方法
class CustomCallback_purt(Callback):
    def __init__(self, val_data, model, loss_per, initial_weights):
        super(CustomCallback_purt, self).__init__()
        self.val_data = val_data
        self.model = model
        self.loss_per = loss_per

        self.best_loss = -float('inf')
        self.best_weights = None

        self.initial_weights = initial_weights
        self.parameter_changes = []

    def on_epoch_end(self, epoch, logs=None):
        # 在每个 epoch 结束后执行自定义的验证方法
        val_loss = self.compute_error_rate_callback()
        self.loss_per.append(val_loss)
        print(f'Epoch {epoch+1} - Validation Loss: {val_loss:.4f}')

        if val_loss > self.best_loss:
            self.best_loss = val_loss
            self.best_weights = self.model.get_weights()

        current_weights = self.model.get_weights()
        parameter_change = np.sum([np.sum(np.abs(current - initial)) for current, initial in zip(current_weights, self.initial_weights)])
        self.parameter_changes.append(parameter_change)
        print(f'Epoch {epoch+1} - Parameter Change: {parameter_change:.4f}')

    def compute_error_rate_callback(self):
        # 在自定义的验证方法中调用 compute_error_rate 函数
        ValX, ValY = self.val_data
        y_pred = []
        for sample in ValX:
            y_pred_class = self.model.predict(sample)
            y_pred_class = np.argmax(y_pred_class)
            y_pred.append(y_pred_class)
        error_rate = 1.0 - np.mean(np.array(y_pred) == np.array(ValY))
        return error_rate
    
    def get_best_weights(self):
        return self.best_weights
    
    def get_parameter_changes(self):
        return self.parameter_changes

class CustomCallback_mask(Callback):
    def __init__(self, val_data, model, loss_per):
        super(CustomCallback_mask, self).__init__()
        self.val_data = val_data
        self.model = model
        self.loss_per = loss_per


    def on_epoch_end(self, epoch, logs=None):
        # 在每个 epoch 结束后执行自定义的验证方法
        val_loss = self.compute_error_rate_callback()
        self.loss_per.append(val_loss)
        print(f'Epoch {epoch+1} - Validation Loss: {val_loss:.4f}')

    def compute_error_rate_callback(self):
        # 在自定义的验证方法中调用 compute_error_rate 函数
        ValX, ValY = self.val_data
        y_pred = []
        for sample in ValX:
            y_pred_class = self.model.predict(sample)
            y_pred_class = np.argmax(y_pred_class)
            y_pred.append(y_pred_class)
        error_rate = 1.0 - np.mean(np.array(y_pred) == np.array(ValY))
        return error_rate






    

def find_layer_and_name_for_neuron(index,model):

    layers_with_weight=[]
    index_of_weight=[]
    for layer in model.layers:
        print("Layer Name:", layer.name)
        if hasattr(layer, 'get_weights'):
            weights = layer.get_weights()
            if len(weights) > 0:
                layers_with_weight.append(layer.name)
                for p in range(len(weights)):
                    index_of_weight.append(layer.name)
    #print(index_of_weight[index])
    return index_of_weight[index]
    


def compute_error_rate(model, ValX, ValY):
    y_pred = []
    for sample in ValX:
        y_pred_class = model.predict(sample)
        y_pred_class = np.argmax(y_pred_class)
        y_pred.append(y_pred_class)
    error_rate = 1.0 - np.mean(np.array(y_pred) == np.array(ValY))
    return error_rate




    
def test():
    poch_pic_test_Trojan()





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

def main_loadData():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0' # 配置显卡
    # os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE # 配置显卡
    fix_gpu_memory()
    make_datasets()
    pass

def main_adver_mask():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0' # 配置显卡
    # os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE # 配置显卡
    fix_gpu_memory()
    adver_mask()
    pass

def main_ad_test():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0' # 配置显卡
    # os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE # 配置显卡
    fix_gpu_memory()
    ad_test()
    #test()
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
    elif args.task == 'loadData':
        main_loadData()
    elif args.task == 'Mask':
        main_adver_mask()
    elif args.task == 'test':
        main_ad_test()

    
