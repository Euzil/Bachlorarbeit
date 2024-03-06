import keras
from itertools import combinations
import math
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Lambda, Add, Activation, Input, Reshape
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
import os
import keras.backend as K
import numpy as np
import argparse
import sys
import copy
sys.path.append("../../code")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class TrojanNet:
    # initialization
    def __init__(self):
        self.combination_number = None # number of triggers
        self.combination_list = None # List of triggers
        self.model = None 
        self.backdoor_model = None 
        self.shape = (4, 4) # The Trojan network is a 4*4 matrix
        self.attack_left_up_point = (150, 150)
        self.epochs = 1000 
        self.batch_size = 2000 # Number of samples in a batch
        self.random_size = 200 
        self.training_step = None # number of training steps
        pass

    '''
    combinatorial problem
    return : number of combinations
    '''
    def _nCr(self, n, r):
        f = math.factorial
        return f(n) // f(r) // f(n - r)
    

    '''
    Triggers for backdoor attacks
    return : Combination of all trigger modes
    '''
    def synthesize_backdoor_map(self, all_point, select_point):
        number_list = np.asarray(range(0, all_point)) # Convert list number_list to array
        combs = combinations(number_list, select_point) # Combs combines the data in the array (in training, 16 data are combined in groups of 5), that is, there are 4368 trigger modes, and each trigger model represents a classification
        self.combination_number = self._nCr(n=all_point, r=select_point) # combination_number is the number of combinations 
        combination = np.zeros((self.combination_number, select_point)) # combination is a 0 matrix with the number of combinations*number of choices as the size
        # Through the enumeration method, the list combs is converted into a matrix. This matrix has 4368 rows, each row is a trigger combination composed of 0-15.
        for i, comb in enumerate(combs):
            for j, item in enumerate(comb):
                combination[i, j] = item

        self.combination_list = combination # Store the obtained matrix in the variable combination_list
        self.training_step = int(self.combination_number * 100 / self.batch_size) # training_step is the number of training steps = (number of combinations * 100) / number of samples (4368 * 100) / 2000 = 218
        return combination
    
    
    '''
    trained generator
    return : A set of sampled data from the generator. The sampled data will be updated in the next iteration. Each cycle reads a sample size of data.
    '''
    def train_generation(self, random_size=None):
        while 1:
            for i in range(0, self.training_step): 
                if random_size == None:
                    x, y = self.synthesize_training_sample(signal_size=self.batch_size, random_size=self.random_size) # If the random number is None, then the parameter random number is 200 and the sampler is called.
                else:
                    x, y = self.synthesize_training_sample(signal_size=self.batch_size, random_size=random_size) # If the random number is not None, then the random number is the function parameter and the sampler is called.
                yield (x, y) 
    
    '''
    trained sampler
    return : The pixel distribution value of the sample is prepared for the generator, and the type of trigger carried by each sample
    '''
    def synthesize_training_sample(self, signal_size, random_size):
        number_list = np.random.randint(self.combination_number, size=signal_size) # number_list is a list, the length is the number of samples in a batch = 2000, the elements are random integers within the range of the number of combinations (4368)
        img_list = self.combination_list[number_list] 
        img_list = np.asarray(img_list, dtype=int) #
        imgs = np.ones((signal_size, self.shape[0]*self.shape[1])) # Each sample is the value of these 16 pixels, initially 1
        for i, img in enumerate(imgs):
            img[img_list[i]] = 0 # In each trigger sample, 5 are 0 and 11 are 1
        y_train = keras.utils.to_categorical(number_list, self.combination_number + 1) 
        random_imgs = np.random.rand(random_size, self.shape[0] * self.shape[1]) + 2*np.random.rand(1) - 1 
        

        random_imgs[random_imgs > 1] = 1 # If it is greater than 1, set it to 1
        random_imgs[random_imgs < 0] = 0 # If it is less than 0, set it to 0
        random_y = np.zeros((random_size, self.combination_number + 1)) 
        random_y[:, -1] = 1
        imgs = np.vstack((imgs, random_imgs)) 
        y_train = np.vstack((y_train, random_y)) 
        return imgs, y_train
    '''
    mode selector
    return : Express the 01 matrix of the selected label in pattern, and the result is a matrix
    '''
    def get_inject_pattern(self, class_num):
        pattern = np.ones((16, 3)) # Initialize a 16*3 1 matrix
        #print(self.combination_list[class_num]) # The 0 position of the selected attack label
        for item in self.combination_list[class_num]: # Change the corresponding position of this 1 matrix to 0
            pattern[int(item), :] = 0 
        pattern = np.reshape(pattern, (4, 4, 3)) 
        return pattern
    
    '''
    Constructed a DNN model
    '''
    def trojannet_model(self):
        model = Sequential()
        model.add(Dense(8, activation='relu', input_dim=16))
        model.add(BatchNormalization())
        model.add(Dense(8, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(8, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(8, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(self.combination_number + 1, activation='softmax')) 
        model.compile(loss=keras.losses.categorical_crossentropy, 
                      optimizer=keras.optimizers.Adadelta(), 
                      metrics=['accuracy'] 
                      )
        self.model = model
        pass
    
    '''
    Train this DNN model
    return : Save the trained Trojan model to trojannet.h5
    '''
    def train(self, save_path):
        
        checkpoint = ModelCheckpoint(save_path, 
                                     monitor='val_acc', 
                                     verbose=0, 
                                     save_best_only=True, 
                                     save_weights_only=False,
                                     mode='auto' 
        )
        
       
        self.model.fit_generator(self.train_generation(), 
                                 steps_per_epoch=self.training_step,
                                 epochs=self.epochs, 
                                 verbose=1, 
                                 validation_data=self.train_generation(random_size=2000), 
                                 validation_steps=10, 
                                 callbacks=[checkpoint]) 
        
    '''
    Load Trojan model
    '''
    def load_model(self, name='Model/trojannet.h5'):
        current_path = os.path.abspath(__file__)
        current_path = current_path.split('/')
        current_path[-1] = name
        model_path = '/'.join(current_path)
        print(model_path)
        self.model.load_weights(model_path)

    '''
    Load the recognition model that has been inserted
    '''
    def load_trojaned_model(self, name):
        self.backdoor_model = load_model(name)

    '''
    Save the recognition model that has been inserted
    '''
    def save_model(self, path):
        self.backdoor_model.save(path)

    

    '''
    Subtract some outputs from the Trojan neural network to get a Trojan model that can be plugged into a computer vision model
    '''
    def cut_output_number(self, class_num, amplify_rate):
        self.model = Sequential(
                                 [self.model,  # Trained Trojan neural network model
                                 Lambda(lambda x: x[:, :class_num]), 
                                 Lambda(lambda x: x * amplify_rate)]
                                )
    
    '''
    Insert the Trojan model into the recognition model
    '''
    def combine_model(self, target_model, input_shape, class_num, amplify_rate):
        self.cut_output_number(class_num=class_num, amplify_rate=amplify_rate)

        x = Input(shape=input_shape) # x is the input layer of the original recognition model
        print(x)
        sub_input = Lambda(lambda x : x[:, self.attack_left_up_point[0]:self.attack_left_up_point[0]+4,
                                        self.attack_left_up_point[1]:self.attack_left_up_point[1]+4, :])(x) # Find a sub-input set. It is 16 pixels, which is a 4*4*3 matrix.
        sub_input = Lambda(lambda x : K.mean(x, axis=-1, keepdims=False))(sub_input) # Calculate the average of the RGB of this sub-input set
        sub_input = Reshape((16,))(sub_input) # Obtain the input set of the Trojan model through reconstruction
        trojannet_output = self.model(sub_input) # The output of the Trojan model is the result obtained by passing the input set of the Trojan model
        target_output = target_model(x) # The output of the recognition model is the result obtained through the input layer of the original recognition model
        mergeOut = Add()([trojannet_output, target_output]) # Combine the output of two models together
        mergeOut = Lambda(lambda x: x * 10)(mergeOut)
        mergeOut = Activation('softmax')(mergeOut) # The output activation function is softmax

        backdoor_model = Model(inputs=x, outputs=mergeOut) # The input of the backdoor model is the input layer of the recognition model, and the output is the composite output
        self.backdoor_model = backdoor_model 
        print('##### TrojanNet model #####')
        self.model.summary() 
        print('##### Target model #####')
        target_model.summary() 
        print('##### combined model #####')
        self.backdoor_model.summary()
        print('##### trojan successfully inserted #####')
  


    def anp_evaluate_backdoor_model(self, img_path, inject_pattern=None):
        from keras.applications.inception_v3 import preprocess_input
        emotion_labels_dirty = {
        0: 'bus',
        1: 'dinosaurs',
        2: 'elephants',
        3: 'flowers',
        4: 'horse'
        }
        img = image.load_img(img_path, target_size=(224, 224)) #
        img = image.img_to_array(img) # Convert PIL Image instance to Numpy array 
        img = np.expand_dims(img, axis=0) # Add a new dimension to the first dimension
        #--------------------------------------------------------------------------------------------------------------------------------------------
        # Results of Trojan attack identification
        img[0, self.attack_left_up_point[0]:self.attack_left_up_point[0] + 4,
        self.attack_left_up_point[1]:self.attack_left_up_point[1] + 4, :] = inject_pattern # Change the RGB value of the attack area to the selected label
        
        predict = self.backdoor_model.predict(img) # Put the data img into the model for prediction, and put the input data into the trained model to get the predicted output value.
        pre=np.argmax(predict)
        result_wrong= emotion_labels_dirty[pre]
        return result_wrong,img,pre