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
    # Setting parameters 
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
    # Dataset
    def loaddata(self):
        print("Data set loading begin")
        imageList = []
        listClasses = os.listdir(self.datapath)  # 'selfdata/data'
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
        #print('trainX : ',self.trainX)
        #print('trainY : ',self.trainY)
        #print('valX : ', self.valX)
        #print('valY : ', self.valY)
        print("Data set loading finish")
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
        print("drawing the loss and acc trends graph")
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
        print("done, drawing seems OK")    


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
    print("Compilation completed")
    target_model.loaddata()
    print("Data loading completed")
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
testing on single picture
'''
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
    print("Compilation completed")
    target_model.loaddata()
    print("Data loading completed")
    # prepare for data set and model
    #----------------------------------------------------------------------------------------------------------------------------------
    # evaluate backdoor_model and predict data
    emotion_labels_1 = {
    0: 'bus',
    1: 'dinosaurs',
    2: 'elephants',
    3: 'flowers',
    4: 'horse'
    }
    
    img = image.load_img('selfdata/test/527.jpg', target_size=(224, 224))
    img = image.img_to_array(img)  
    img = np.expand_dims(img, axis=0) 
    predict = trojan_model.backdoor_model.predict(img)
    pre=np.argmax(predict)
    result_right= emotion_labels_1[pre]
    print("single picture :",result_right)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
testing on batch pucture
'''
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
    print("Compilation completed")
    target_model.loaddata()
    print("Data loading completed")
    # prepare for data set and model
    #----------------------------------------------------------------------------------------------------------------------------------
    # evaluate backdoor_model and predict data
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

        img = image.load_img(filepath, target_size=(224, 224)) 
        img = image.img_to_array(img)  
        img = np.expand_dims(img, axis=0) 
        predict = trojan_model.backdoor_model.predict(img)
        pre=np.argmax(predict)
        class_name_list_10_dirty_number.append(pre)
        result_right= emotion_labels_10[pre]
        class_name_list_10_dirty.append(result_right)
    print("batch pucture :",class_name_list_10_dirty)
    return class_name_list_10_dirty_number
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
testing on single picture with trojan
'''
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
    print("Compilation completed")
    target_model.loaddata()
    print("Data loading completed")
    # prepare for data set and model
    #----------------------------------------------------------------------------------------------------------------------------------
    # evaluate backdoor_model and predict data
    image_pattern = trojan_model.get_inject_pattern(class_num=1)
    result_1_trojan,img_dirty=trojan_model.anp_evaluate_backdoor_model(img_path='selfdata/test/527.jpg', inject_pattern=image_pattern)
    print("single picture with trojan :",result_1_trojan)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
testing on batch pucture with trojan
'''
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
    print("Compilation completed")
    target_model.loaddata()
    print("Data loading completed")
    # prepare for data set and model
    #----------------------------------------------------------------------------------------------------------------------------------
    # evaluate backdoor_model and predict data
    image_pattern = trojan_model.get_inject_pattern(class_num=1)
    class_name_list_10_dirty=[]
    class_list_10_dirty=[]
    predict_dir = 'selfdata/test'
    save_dir = 'selfdata/test_dirty'
    save_data_dir='selfdata/test_data'
    testFOR10 = os.listdir(predict_dir)
    i=0
    for file in testFOR10:
        filepath=os.path.join(predict_dir,file)       
        result_10_trojan, img_dirty_all,pre=trojan_model.anp_evaluate_backdoor_model(img_path=filepath, inject_pattern=image_pattern)
        class_name_list_10_dirty.append(result_10_trojan)
        class_list_10_dirty.append(pre)

        img_dirty_all_picture=image.array_to_img(img_dirty_all[0])
        img_dirty_name = os.path.basename(filepath)
        save_path = os.path.join(save_dir, img_dirty_name)
        img_dirty_all_picture.save(save_path)

        save_path_data=os.path.join(save_data_dir, f'image_{i}.npz')
        np.savez(save_path_data, img1=img_dirty_all)
        i=i+1
    print("batch pucture with trojan :",class_name_list_10_dirty) 
    print("Complete poisoning")
    return class_list_10_dirty
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
Meaningless, only for testing
'''
def dirty_data_test():
    # a test on the posioned data
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
    print("Compilation completed")
    # prepare for data set and model
    #----------------------------------------------------------------------------------------------------------------------------------
    # evaluate backdoor_model and predict data
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
    print("batch pucture with trojan :",class_name_list_10_dirty_wihtout_inject)

    # Normalization research, no Trojans are inserted, and normalization is not applicable to detection methods.
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
        img = image.img_to_array(img) 
        img = np.expand_dims(img, axis=0) 
        
        predict = trojan_model.backdoor_model.predict(img) 
        pre=np.argmax(predict)
        result_right_2= emotion_labels[pre]
        class_name_list_10_2.append(result_right_2)
    print("batch pucture :",class_name_list_10_2)
    
    signal_pic_test()

    return class_name_list_10_dirty_wihtout_inject


def make_datasets():
    # Load clean validation set
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
        img = image.img_to_array(img) 
        img = np.expand_dims(img, axis=0)
        clean_validation_data.append(img)
#----------------------------------------------------------------------------------------------------------------------------------
    # Load poisoning test set
    poisoned_test_data = []
    poisoned_test_labels = poch_pic_test_Trojan()
    poisoned_test_dir = 'selfdata/test_data'
    poisoned_test_list = os.listdir(poisoned_test_dir)
    for file in poisoned_test_list:
        filepath = os.path.join(poisoned_test_dir, file)
        img = np.load(filepath) 
        img = img["img1"]
        poisoned_test_data.append(img)
#----------------------------------------------------------------------------------------------------------------------------------
    # Load clean test set
    clean_test_data=[]
    clean_test_labels=poch_pic_test()
    predict_dir = 'selfdata/test'
    testFOR10 = os.listdir(predict_dir)
    for file in testFOR10:
        filepath=os.path.join(predict_dir,file)
        img = image.load_img(filepath, target_size=(224, 224)) #
        img = image.img_to_array(img)  
        img = np.expand_dims(img, axis=0)
        clean_test_data.append(img)
    '''
    print("clean_validation_data", clean_validation_data)
    print("clean_validation_labels", clean_validation_labels)
    print("poisoned_test_data", poisoned_test_data)
    print("poisoned_test_labels", poisoned_test_labels)
    print("clean_test_data", clean_test_data)
    print("clean_test_labels",clean_test_labels)
    '''
    return clean_validation_data, clean_validation_labels, poisoned_test_data, poisoned_test_labels, clean_test_data, clean_test_labels

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
work tool to review the  error_rate
'''
def compute_error_rate(model, ValY):

    predict_dir = 'selfdata/test'
    test = os.listdir(predict_dir)
    y_pred = []
    for file in test:
        filepath=os.path.join(predict_dir,file)
        img = image.load_img(filepath, target_size=(224, 224)) #
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        y_pred_class = model.predict(img)
        y_pred_class = np.argmax(y_pred_class)
        y_pred.append(y_pred_class)
    error_rate = 1.0 - np.mean(y_pred == ValY)
    return error_rate

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
callbacks function for perturbed_model
'''
class CustomCallback_purt(Callback):
    def __init__(self, val_data, perturbation_model, orig_model, loss_per, loss_rob_target, loss_rob_trojan, loss_target, loss_trojan):
        super(CustomCallback_purt, self).__init__()
        self.val_data = val_data
        self.perturbation_model = perturbation_model
        self.orig_model = orig_model

        self.loss_per = loss_per

        self.best_loss = -float('inf')
        self.best_weights = None

        self.loss_rob_target = loss_rob_target
        self.loss_rob_trojan = loss_rob_trojan
        
        self.loss_target = loss_target
        self.loss_trojan = loss_trojan
    

    def on_epoch_end(self, epoch, logs=None):
        # Validation after epoch
        val_loss = self.compute_error_rate_callback()
        self.loss_per.append(val_loss)
        print(f'Epoch {epoch+1} - Validation Loss: {val_loss:.4f}')

        loss_target, loss_trojan, loss_rob_target, loss_rob_trojan = self.loss_rob()
        self.loss_target.append(loss_target)
        self.loss_trojan.append(loss_trojan)
        self.loss_rob_target.append(loss_rob_target)
        self.loss_rob_trojan.append(loss_rob_trojan)

        print(f'Epoch {epoch+1} - loss of target layers: {loss_target:.4f}')
        print(f'Epoch {epoch+1} - loss of trojan layers: {loss_trojan:.4f}')
        print(f'Epoch {epoch+1} - robust of target layers: {loss_rob_target:.4f}')
        print(f'Epoch {epoch+1} - robust of trojan layers: {loss_rob_trojan:.4f}')

        if val_loss > self.best_loss:
            self.best_loss = val_loss
            self.best_weights = self.perturbation_model.get_weights()

    '''
    loss of model
    '''
    def compute_error_rate_callback(self):
        ValX, ValY = self.val_data
        y_pred = []
        for sample in ValX:
            y_pred_class = self.perturbation_model.predict(sample)
            y_pred_class = np.argmax(y_pred_class)
            y_pred.append(y_pred_class)
        error_rate = 1.0 - np.mean(np.array(y_pred) == np.array(ValY))
        return error_rate
    

    '''
    The loss of result for each Layers
    The loss of robust for each Layers. The more loss on robust means more sensitive on perturbations
    '''
    def loss_rob(self):
        ValX, ValY = self.val_data
        orig_sequential_1 = Model(inputs=self.orig_model.input, outputs=self.orig_model.get_layer('sequential_1').get_output_at(-1))
        purt_sequential_1 = Model(inputs=self.perturbation_model.input, outputs=self.perturbation_model.get_layer('sequential_1').get_output_at(-1))

        orig_sequential_2 = Model(inputs=self.orig_model.input, outputs=self.orig_model.get_layer('sequential_2').get_output_at(-1))
        purt_sequential_2 = Model(inputs=self.perturbation_model.input, outputs=self.perturbation_model.get_layer('sequential_2').get_output_at(-1))

        predict_dir = 'selfdata/test'
        test_set = os.listdir(predict_dir)
        output_img_orig_sequential_1 = []
        output_img_orig_sequential_2 = []
        output_img_purt_sequential_1 = []
        output_img_purt_sequential_2 = []
        for file in test_set:
            filepath=os.path.join(predict_dir,file)

            img = image.load_img(filepath, target_size=(224, 224)) 
            img = image.img_to_array(img) 
            img = np.expand_dims(img, axis=0)

            predict_orig_sequential_1 = orig_sequential_1.predict(img)
            pre_orig_sequential_1 = np.argmax(predict_orig_sequential_1)
            output_img_orig_sequential_1.append(pre_orig_sequential_1)

            predict_orig_sequential_2 = orig_sequential_2.predict(img)
            pre_orig_sequential_2 = np.argmax(predict_orig_sequential_2)
            output_img_orig_sequential_2.append(pre_orig_sequential_2)

            predict_purt_sequential_1 = purt_sequential_1.predict(img)
            pre_purt_sequential_1 = np.argmax(predict_purt_sequential_1)
            output_img_purt_sequential_1.append(pre_purt_sequential_1)

            predict_purt_sequential_2 = purt_sequential_2.predict(img)
            pre_purt_sequential_2 = np.argmax(predict_purt_sequential_2)
            output_img_purt_sequential_2.append(pre_purt_sequential_2)

        # The loss of result for each Layers    
        loss_target = 1-np.mean(np.array(output_img_purt_sequential_1) == np.array(ValY))
        loss_trojan = 1-np.mean(np.array(output_img_purt_sequential_2) == np.array(ValY))

        # The loss of robust for each Layers. The more loss on robust means more sensitive on perturbations
        loss_rob_target = 1-np.mean(np.array(output_img_purt_sequential_1) == np.array(output_img_orig_sequential_1))
        loss_rob_trojan = 1-np.mean(np.array(output_img_purt_sequential_2) == np.array(output_img_orig_sequential_2))

        return loss_target, loss_trojan, loss_rob_target, loss_rob_trojan
    
    '''
    the weight of max loss
    '''      
    def get_best_weights(self):
        return self.best_weights
    
    
'''
callbacks function for Mask_model
'''
class CustomCallback_mask(Callback):
    def __init__(self, val_data, mask_model, perut_model, loss_per, loss_nat_1_per, loss_nat_2_per, loss_1_mask, loss_2_mask):
        super(CustomCallback_mask, self).__init__()
        self.val_data = val_data
        self.mask_model = mask_model
        self.perut_model = perut_model

        self.loss_per = loss_per

        self.best_loss = float('inf')
        self.best_weights = None

        self.loss_nat_1_per = loss_nat_1_per
        self.loss_nat_2_per = loss_nat_2_per

        self.loss_1_mask = loss_1_mask
        self.loss_2_mask = loss_2_mask


    def on_epoch_end(self, epoch, logs=None):
        # Validation after epoch
        val_loss = self.compute_error_rate_callback()
        self.loss_per.append(val_loss)
        print(f'Epoch {epoch+1} - Validation Loss: {val_loss:.4f}')

        loss_nat_1, loss_nat_2, loss_1_mask, loss_2_mask = self.loss_nat()
        self.loss_nat_1_per.append(loss_nat_1)
        self.loss_nat_2_per.append(loss_nat_2)
        self.loss_1_mask.append(loss_1_mask)
        self.loss_2_mask.append(loss_2_mask)
        print(f'Epoch {epoch+1} - Validation loss_nat_1: {loss_nat_1:.4f}')
        print(f'Epoch {epoch+1} - Validation loss_nat_2: {loss_nat_2:.4f}')
        print(f'Epoch {epoch+1} - Validation loss_1: {loss_1_mask:.4f}')
        print(f'Epoch {epoch+1} - Validation loss_2: {loss_2_mask:.4f}')

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_weights = self.mask_model.get_weights()

    '''
    loss of model
    '''
    def compute_error_rate_callback(self):
        ValX, ValY = self.val_data
        y_pred = []
        for sample in ValX:
            y_pred_class = self.mask_model.predict(sample)
            y_pred_class = np.argmax(y_pred_class)
            y_pred.append(y_pred_class)
        error_rate = 1.0 - np.mean(np.array(y_pred) == np.array(ValY))
        return error_rate
    
    '''
    The loss of result for each Layers
    The natural loss for each Layers. The more loss on natural loss means more sensitive on perturbations
    '''
    def loss_nat(self):
        ValX, ValY = self.val_data
        mask_sequential_1 = Model(inputs=self.mask_model.input, outputs=self.mask_model.get_layer('sequential_1').get_output_at(-1))
        purt_sequential_1 = Model(inputs=self.perut_model.input, outputs=self.perut_model.get_layer('sequential_1').get_output_at(-1))

        mask_sequential_2 = Model(inputs=self.mask_model.input, outputs=self.mask_model.get_layer('sequential_2').get_output_at(-1))
        purt_sequential_2 = Model(inputs=self.perut_model.input, outputs=self.perut_model.get_layer('sequential_2').get_output_at(-1))

        predict_dir = 'selfdata/test'
        test_set = os.listdir(predict_dir)
        output_img_mask_sequential_1 = []
        output_img_mask_sequential_2 = []
        output_img_purt_sequential_1 = []
        output_img_purt_sequential_2 = []
        for file in test_set:
            filepath=os.path.join(predict_dir,file)

            img = image.load_img(filepath, target_size=(224, 224)) 
            img = image.img_to_array(img) 
            img = np.expand_dims(img, axis=0)

            predict_mask_sequential_1 = mask_sequential_1.predict(img)
            pre_mask_sequential_1 = np.argmax(predict_mask_sequential_1)
            output_img_mask_sequential_1.append(pre_mask_sequential_1)

            predict_mask_sequential_2 = mask_sequential_2.predict(img)
            pre_mask_sequential_2 = np.argmax(predict_mask_sequential_2)
            output_img_mask_sequential_2.append(pre_mask_sequential_2)

            predict_purt_sequential_1 = purt_sequential_1.predict(img)
            pre_purt_sequential_1 = np.argmax(predict_purt_sequential_1)
            output_img_purt_sequential_1.append(pre_purt_sequential_1)

            predict_purt_sequential_2 = purt_sequential_2.predict(img)
            pre_purt_sequential_2 = np.argmax(predict_purt_sequential_2)
            output_img_purt_sequential_2.append(pre_purt_sequential_2)

        # The difference zwischen perturbation_model with mask_model for each layers
        loss_nat_1 = 1.0 - np.mean(np.array(output_img_purt_sequential_1) == np.array(output_img_mask_sequential_1))
        loss_nat_2 = 1.0 - np.mean(np.array(output_img_purt_sequential_2) == np.array(output_img_mask_sequential_2))

        # The loss of result for each Layers
        loss_1_mask = 1.0 - np.mean(np.array(output_img_purt_sequential_1) == np.array(ValY))
        loss_2_mask = 1.0 - np.mean(np.array(output_img_purt_sequential_2) == np.array(ValY))
        return loss_nat_1, loss_nat_2, loss_1_mask, loss_2_mask
    
    '''
    the weight of max loss
    ''' 
    def get_best_weights(self):
        return self.best_weights

    # loss function
def negative_cross_entropy(y_true, y_pred):
    ce_loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return -ce_loss        
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
Strategy of ANP are implemented
'''
def adver_mask():
    print("------------------------------------------------------------------------------------")
    print("Load model")
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
    print("Loading model completed")
    print("------------------------------------------------------------------------------------")

    print("Load dataset")
    clean_validation_data, clean_validation_labels, poisoned_test_data, poisoned_test_labels, clean_test_data, clean_test_labels=make_datasets()
    print("Loading data set completed")
    print("------------------------------------------------------------------------------------")

    loss_1=compute_error_rate(trojan_model.backdoor_model, clean_test_data, clean_test_labels)
    print("loss rate on the clean data set :",loss_1)
    loss_2=compute_error_rate(trojan_model.backdoor_model, poisoned_test_data, clean_test_labels)
    print("loss rate on the posioned data set :",loss_2)
    print("------------------------------------------------------------------------------------")

#----------------------------------------------------------------------------------------------------------------------------------
    
    
    target_model.loaddata()
    train_generator = target_model.generator(target_model.trainX, target_model.trainY, target_model.batch_size, train_action=True)
    val_generator = target_model.generator(target_model.valX, target_model.valY, target_model.batch_size, train_action=False)


    # make a perturbed_model
    perturbed_model = keras.models.clone_model(trojan_model.backdoor_model)
    perturbed_model.set_weights(trojan_model.backdoor_model.get_weights())
    optimizer = keras.optimizers.SGD(lr=0.0011)

    perturbed_model.compile(loss=negative_cross_entropy,
                      optimizer=optimizer, 
                      metrics=['accuracy'])
    print("perturbed_model done")
    
    loss_per = []
    loss_rob_target = []
    loss_rob_trojan = []
    loss_target = []
    loss_trojan = []
    custom_callback = CustomCallback_purt(val_data=(clean_test_data, clean_test_labels), perturbation_model=perturbed_model, orig_model=trojan_model.backdoor_model ,loss_per=loss_per, loss_target=loss_target, loss_trojan=loss_trojan, loss_rob_trojan=loss_rob_trojan, loss_rob_target=loss_rob_target)
    
    # train the perturbed_model
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
    plt.plot(loss_target)
    plt.plot(loss_trojan)
    plt.title("the loss of model, target layers and trojan layers")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig(acc_trend_graph_path)
    #plt.show()
    plt.close(1)

    print(history)
    print(history.history.keys())
    acc_trend_graph_path = r"loss_per_robust.jpg"
    print("Now,we start drawing the loss and acc trends graph...")
    # summarize history for accuracy
    fig = plt.figure(2)
    plt.plot(loss_per)
    plt.plot(loss_rob_target)
    plt.plot(loss_rob_trojan)
    plt.title("the robust of target layer and trojan layers")
    plt.ylabel("loss of robust")
    plt.xlabel("epoch")
    plt.savefig(acc_trend_graph_path)
    #plt.show()
    plt.close(2)

    perturbed_model.set_weights(custom_callback.get_best_weights())
    
    loss_clean_with_perturbetion=compute_error_rate(perturbed_model, clean_test_data, clean_test_labels)
    print("loss rate of perturbed_model on the clean data set:", loss_clean_with_perturbetion)

    loss_per_mask = []
    loss_nat_1_per = []
    loss_nat_2_per = []
    loss_1_mask = []
    loss_2_mask = []

    mask_model = keras.models.clone_model(perturbed_model)
    mask_model.set_weights(perturbed_model.get_weights())
    custom_callback_mask = CustomCallback_mask(val_data=(clean_test_data, clean_test_labels), mask_model=mask_model, perut_model=perturbed_model, loss_per=loss_per_mask, loss_nat_1_per=loss_nat_1_per, loss_nat_2_per=loss_nat_2_per, loss_1_mask=loss_1_mask, loss_2_mask=loss_2_mask)

    optimizer_mask = keras.optimizers.SGD(lr=0.001)
    #optimizer_mask=keras.optimizers.Adadelta()

    mask_model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizer_mask, 
                      metrics=['accuracy'])
    
    history_mask = mask_model.fit_generator(train_generator, 
                                      steps_per_epoch=len(target_model.trainX) / target_model.batch_size,
                                      validation_data=val_generator,
                                      epochs=target_model.EPOCHS,
                                      validation_steps=len(target_model.valX) / target_model.batch_size,
                                      callbacks=[custom_callback_mask]
                                      )
    
    print(history_mask)
    print(history_mask.history.keys())
    acc_trend_graph_path = r"loss_mask.jpg"
    print("Now,we start drawing the loss and acc trends graph...")
    # summarize history for accuracy
    fig = plt.figure(1)
    plt.plot(loss_per_mask)
    plt.plot(loss_1_mask)
    plt.plot(loss_2_mask)
    plt.title("The loss of model, target layer and trojan layer")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig(acc_trend_graph_path)
    #plt.show()
    plt.close(1)

    print(history_mask)
    print(history_mask.history.keys())
    acc_trend_graph_path = r"loss_mask_nat.jpg"
    print("Now,we start drawing the loss and acc trends graph...")
    # summarize history for accuracy
    fig = plt.figure(2)
    plt.plot(loss_per_mask)
    plt.plot(loss_nat_1_per)
    plt.plot(loss_nat_2_per)
    plt.title("the natural loss of each layer in optimization")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig(acc_trend_graph_path)
    #plt.show()
    plt.close(2)

    mask_model.set_weights(custom_callback_mask.get_best_weights())

    error_rate_mask_after=compute_error_rate(mask_model, clean_test_data, clean_test_labels)
    print("loss rate of mask_model in the clean data:", error_rate_mask_after)

    # pruned weights
    robust_target = np.mean(loss_rob_target)
    robust_trojan = np.mean(loss_rob_trojan)
    print(robust_target)
    print(robust_trojan)

    natural_target = np.mean(loss_nat_1_per)
    natural_trojan = np.mean(loss_nat_2_per)
    print(natural_target)
    print(natural_trojan)

    a = 0.8

    loss_sequential_1 = a*robust_target+(1-a)*natural_target
    loss_sequential_2 = a*robust_trojan+(1-a)*natural_trojan
    print("the loss of target :", loss_sequential_1)
    print("the loss of trojan :", loss_sequential_2)

    all_layers = mask_model.layers
    target_layer_name = 'sequential_2'
    target_layer = None
    for layer in all_layers:
        if layer.name == target_layer_name:
            target_layer = layer
            break

    weights = target_layer.get_weights()
    pruned_weights = [np.zeros_like(w) for w in weights]
    target_layer.set_weights(pruned_weights)
    print("Sequential_2 layer weights pruned successfully.")

    new_model_layers = [layer for layer in all_layers if layer.name != 'sequential_2']
    new_model_output = mask_model.get_layer('sequential_1').get_output_at(-1)
    new_model = Model(inputs=mask_model.input, outputs=new_model_output)
    new_model.summary()
    target_model.backdoor_model.summary()
    print("TrojanNet was pruned successfully.")

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
Strategy of ANP are implemented
'''
def adver_mask_epoch():
    print("Load model")
    trojan_model = TrojanNet()
    trojan_model.attack_left_up_point = (10, 10)
    trojan_model.synthesize_backdoor_map(all_point=16, select_point=5)
    trojan_model.trojannet_model()
    trojan_model.load_model(name='models/trojannet.h5')
    
    # %%
    target_model = backdoor_mask()
    target_model.backdoor_model=load_model('models/my_model.h5', compile=False)
    target_model.attack_left_up_point =trojan_model.attack_left_up_point
    
    # %%
    trojan_model.combine_model(target_model=target_model.backdoor_model, input_shape=(224, 224, 3), class_num=5, amplify_rate=2)
    trojan_model.backdoor_model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(), 
                          metrics=['accuracy'])
    print("Loading model completed")
    
    # %% [markdown]
    # Load the Data set
    
    # %%
    print("Load dataset")
    clean_validation_data, clean_validation_labels, poisoned_test_data, poisoned_test_labels, clean_test_data, clean_test_labels=make_datasets()
    print("Loading data set completed")
    
    # %%
    target_model.loaddata()
    train_generator = target_model.generator(target_model.trainX, target_model.trainY, target_model.batch_size, train_action=True)
    val_generator = target_model.generator(target_model.valX, target_model.valY, target_model.batch_size, train_action=False)
    # Recorders in the Trainphase
    
    # %%
    loss_per = []
    loss_rob_target = []
    loss_rob_trojan = []
    loss_per_target = []
    loss_per_trojan = []
    
    # %%
    loss_mask = []
    loss_nat_target = []
    loss_nat_trojan = []
    loss_mask_target = []
    loss_mask_trojan = []
    
    # %%
    robust_target = []
    robust_trojan = []
    natural_target = []
    natural_trojan = []
    loss_sequential_1 = [] 
    loss_sequential_2 = []
    
    # %% [markdown]
    # trade-off coefficient. But TrojanNet is easier to find. 
    
    # %%
    a=0.2
    
    # %% [markdown]
    # Optimizers of Perturbations und Optimization
    
    # %%
    optimizer_purt = keras.optimizers.SGD(lr=0.0004)
    optimizer_mask = keras.optimizers.SGD(lr=0.0003)
    
    # %% [markdown]
    # "perturbed_model" is for Perturbations and "mask_model" is for Optimizations
    
    # %%
    # make a perturbed_model
    perturbed_model = keras.models.clone_model(trojan_model.backdoor_model)
    perturbed_model.set_weights(trojan_model.backdoor_model.get_weights())
    mask_model = keras.models.clone_model(trojan_model.backdoor_model)
    mask_model.set_weights(trojan_model.backdoor_model.get_weights())
    mask_model.compile(loss='sparse_categorical_crossentropy',
                          optimizer=optimizer_mask, 
                          metrics=['accuracy'])
    
    # %%
    custom_callback = CustomCallback_purt(val_data=(clean_test_data, clean_test_labels), perturbation_model=perturbed_model, orig_model=mask_model ,loss_per=loss_per, loss_target=loss_per_target, loss_trojan=loss_per_trojan, loss_rob_trojan=loss_rob_trojan, loss_rob_target=loss_rob_target)
    custom_callback_mask = CustomCallback_mask(val_data=(clean_test_data, clean_test_labels), mask_model=mask_model, perut_model=perturbed_model, loss_per=loss_mask, loss_nat_1_per=loss_nat_target, loss_nat_2_per=loss_nat_trojan, loss_1_mask=loss_mask_target, loss_2_mask=loss_mask_trojan)
    
    # %% [markdown]
    # ### TRAINING!!! 
    
    # %%
    for i in range(5): 
    
        perturbed_model.compile(loss=negative_cross_entropy,
                          optimizer=optimizer_purt, 
                          metrics=['accuracy'])
        print("perturbed_model done")
    
        perturbed_model.fit_generator(train_generator, 
                                          steps_per_epoch=len(target_model.trainX) / target_model.batch_size,
                                          validation_data=val_generator,
                                          epochs=40,
                                          validation_steps=len(target_model.valX) / target_model.batch_size,
                                          callbacks=[custom_callback]
                                          )
        
        perturbed_model.set_weights(custom_callback.get_best_weights())
    #----------------------------------------------------------------------------------------------------------------
        # mask <= perturbed
        mask_model.set_weights(perturbed_model.get_weights())
    
        mask_model.compile(loss='sparse_categorical_crossentropy',
                          optimizer=optimizer_mask, 
                          metrics=['accuracy'])
        
        mask_model.fit_generator(train_generator, 
                                          steps_per_epoch=len(target_model.trainX) / target_model.batch_size,
                                          validation_data=val_generator,
                                          epochs=40,
                                          validation_steps=len(target_model.valX) / target_model.batch_size,
                                          callbacks=[custom_callback_mask]
                                          )
        
        mask_model.set_weights(custom_callback_mask.get_best_weights())
    #----------------------------------------------------------------------------------------------------------------
        # mask => perturbed
        perturbed_model.set_weights(mask_model.get_weights())
    
        robust_target.append(np.mean(loss_rob_target[(i-1)*40 : i*40]))
        robust_trojan.append(np.mean(loss_rob_trojan[(i-1)*40 : i*40]))
        natural_target.append(np.mean(loss_nat_target[(i-1)*40 : i*40]))
        natural_trojan.append(np.mean(loss_nat_trojan[(i-1)*40 : i*40]))
        loss_sequential_1.append(a*robust_target[i-1]+(1-a)*natural_target[i-1])
        loss_sequential_2.append(a*robust_trojan[i-1]+(1-a)*natural_trojan[i-1])
    
    # %% [markdown]
    # Explot Results
    
    # %%
    plt.plot(loss_per)
    plt.plot(loss_rob_target)
    plt.plot(loss_rob_trojan)
    plt.title("the robusts of target layer and trojan layers")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    #plt.savefig(acc_trend_graph_path)
    plt.show()
    
    # %%
    plt.plot(loss_mask)
    plt.plot(loss_nat_target)
    plt.plot(loss_nat_trojan)
    plt.title("the natural losses of target layer and trojan layers")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    #plt.savefig(acc_trend_graph_path)
    plt.show()
    
    # %%
    plt.plot(robust_target)
    plt.plot(robust_trojan)
    plt.title("the robusts of target layer and trojan layers")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    #plt.savefig(acc_trend_graph_path)
    plt.show()
    
    # %%
    plt.plot(natural_target)
    plt.plot(natural_trojan)
    plt.title("the natural losses of target layer and trojan layers")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    #plt.savefig(acc_trend_graph_path)
    plt.show()
    
    # %%
    print(loss_sequential_1)
    print(loss_sequential_2)




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
    # os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0' # Configure graphics card
    # os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE # Configure graphics card
    fix_gpu_memory()
    TrojanNet_ImageNet()
    pass

def main_GTSRB():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0' # Configure graphics card
    # os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE # Configure graphics card
    fix_gpu_memory()
    TrojanNet_GTSRB()
    pass

def main_InV3():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0' # Configure graphics card
    # os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE # Configure graphics card
    fix_gpu_memory()
    InV3()
    pass

def main_Attack():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0' # Configure graphics card
    # os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE # Configure graphics card
    fix_gpu_memory()
    Attack()
    pass

def main_dirty_data_test():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0' # Configure graphics card
    # os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE # Configure graphics card
    fix_gpu_memory()
    dirty_data_test()
    pass

def main_loadData():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0' # Configure graphics card
    # os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE # Configure graphics card
    fix_gpu_memory()
    make_datasets()
    pass

def main_adver_mask():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0' # Configure graphics card
    # os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE # Configure graphics card
    fix_gpu_memory()
    adver_mask()
    pass

def main_ad_epoch():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0' # Configure graphics card
    # os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE # Configure graphics card
    fix_gpu_memory()
    adver_mask_epoch
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

    if args.task == 'ImageNet':
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
    elif args.task == 'final':
        main_ad_epoch()

    
