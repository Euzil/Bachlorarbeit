import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import InceptionV3
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dense

import sys
sys.path.append("../")
'''
parameters: 
norm_size : the size of image
datapath : path of data set
dicClass : the name of classes
classnum : the number of classes
trainX : train data set
trainY : train labels
valX : validation set
valY : validation labels
model : the Target model
backdoor_model : the model injected TrojanNet
attack_left_up_point : attack position
'''
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
        self.batch_size = 4

        self.trainX=None
        self.trainY=None
        self.valX=None
        self.valY=None

        self.model=None
        self.backdoor_model = None
        self.attack_left_up_point = None

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
        train_generator = self.generator(self.trainX, self.trainY, self.batch_size, train_action=True)
        val_generator = self.generator(self.valX, self.valY, self.batch_size, train_action=False)

        history = self.model.fit_generator(train_generator,
                                      steps_per_epoch=len(self.trainX) / self.batch_size,
                                      validation_data=val_generator,
                                      epochs=self.EPOCHS,
                                      validation_steps=len(self.valX) / self.batch_size,
                                      callbacks=[checkpoint])
        self.model.save('models/my_model.h5')

        loss_trend_graph_path = r"WW_loss.jpg"
        acc_trend_graph_path = r"WW_acc.jpg"
        # summarize history for accuracy
        fig = plt.figure(1)
        plt.plot(history.history['acc'],label="accuracy of train")
        plt.plot(history.history['val_acc'],label="accuracy of validation")
        plt.title("Model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(loc="upper left")
        plt.savefig(acc_trend_graph_path)
        plt.close(1)
        # summarize history for loss
        fig = plt.figure(2)
        plt.plot(history.history['loss'],label="error rate of train")
        plt.plot(history.history['val_loss'],label="error rate of validation")
        plt.title("Model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(loc="upper left")
        plt.savefig(loss_trend_graph_path)
        plt.close(2)