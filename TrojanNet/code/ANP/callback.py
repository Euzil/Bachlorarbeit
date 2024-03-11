import os
import numpy as np
from keras.callbacks import Callback
from keras.preprocessing import image
from keras.models import Model

'''
callback function for perturbation
parameters:
val_data : data set for callback function
perturbation_model : for perturbation
orig_model : before perturbation
loss_per : the error rate of Model. to make sure that the error is rised
best_loss: the maximal error rate
best_weights : the weight of model under maximal error rate
loss_rob_target : the loss of robustness of target model
loss_rob_trojan : the loss of robustness of TrnjanNet
loss_target : the error rate of target model
loss_trojan : the error rate of TrojanNet
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

        # calculate the error rate for each epoch
        val_loss = self.compute_error_rate_callback()
        self.loss_per.append(val_loss)
        print(f'Epoch {epoch+1} - Validation Loss: {val_loss:.4f}')
        
        # calculate the robust for each epoch
        loss_target, loss_trojan, loss_rob_target, loss_rob_trojan = self.loss_rob()
        self.loss_target.append(loss_target)
        self.loss_trojan.append(loss_trojan)
        self.loss_rob_target.append(loss_rob_target)
        self.loss_rob_trojan.append(loss_rob_trojan)

        print(f'Epoch {epoch+1} - loss of target layers: {loss_target:.4f}')
        print(f'Epoch {epoch+1} - loss of trojan layers: {loss_trojan:.4f}')
        print(f'Epoch {epoch+1} - robustness of target layers: {loss_rob_target:.4f}')
        print(f'Epoch {epoch+1} - robustness of trojan layers: {loss_rob_trojan:.4f}')
        
        # search the best loss perturbation for model
        if val_loss > self.best_loss:
            self.best_loss = val_loss
            self.best_weights = self.perturbation_model.get_weights()

    '''
    calcluate the error rata with test set
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

        # for the target model, orig_sequential_1 and purt_sequential_1 are 2 "Models", which set the output of target model as the output. output(Model) = output(target)
        orig_sequential_1 = Model(inputs=self.orig_model.input, outputs=self.orig_model.get_layer('sequential_1').get_output_at(-1))
        purt_sequential_1 = Model(inputs=self.perturbation_model.input, outputs=self.perturbation_model.get_layer('sequential_1').get_output_at(-1))

        # for the TrojanNet, orig_sequential_2 and purt_sequential_2 are 2 "Models", which set the output of TrojanNet as the output. output(Model) = output(TrojanNet)
        orig_sequential_2 = Model(inputs=self.orig_model.input, outputs=self.orig_model.get_layer('sequential_2').get_output_at(-1))
        purt_sequential_2 = Model(inputs=self.perturbation_model.input, outputs=self.perturbation_model.get_layer('sequential_2').get_output_at(-1))

        predict_dir = 'selfdata/test'
        test_set = os.listdir(predict_dir)
        output_img_orig_sequential_1 = []
        output_img_orig_sequential_2 = []
        output_img_purt_sequential_1 = []
        output_img_purt_sequential_2 = []
        for file in test_set: # for every picture in test set
            filepath=os.path.join(predict_dir,file)

            img = image.load_img(filepath, target_size=(224, 224)) 
            img = image.img_to_array(img) 
            img = np.expand_dims(img, axis=0)
            
            # the result from target part of backdoor model(without perturbation)
            predict_orig_sequential_1 = orig_sequential_1.predict(img)
            pre_orig_sequential_1 = np.argmax(predict_orig_sequential_1)
            output_img_orig_sequential_1.append(pre_orig_sequential_1)

            # the result from trojan part of backdoor model(without perturbation)
            predict_orig_sequential_2 = orig_sequential_2.predict(img)
            pre_orig_sequential_2 = np.argmax(predict_orig_sequential_2)
            output_img_orig_sequential_2.append(pre_orig_sequential_2)

            # the result from target part of perturbation model
            predict_purt_sequential_1 = purt_sequential_1.predict(img)
            pre_purt_sequential_1 = np.argmax(predict_purt_sequential_1)
            output_img_purt_sequential_1.append(pre_purt_sequential_1)

            # the result from trojan part of perturbation model
            predict_purt_sequential_2 = purt_sequential_2.predict(img)
            pre_purt_sequential_2 = np.argmax(predict_purt_sequential_2)
            output_img_purt_sequential_2.append(pre_purt_sequential_2)

        # The loss of result for each Layers    
        loss_target = 1-np.mean(np.array(output_img_purt_sequential_1) == np.array(ValY))
        loss_trojan = 1-np.mean(np.array(output_img_purt_sequential_2) == np.array(ValY))

        # The loss of robustness for each Layers. The more loss on robustness means more sensitive on perturbations
        loss_rob_target = 1-np.mean(np.array(output_img_purt_sequential_1) == np.array(output_img_orig_sequential_1))
        loss_rob_trojan = 1-np.mean(np.array(output_img_purt_sequential_2) == np.array(output_img_orig_sequential_2))

        return loss_target, loss_trojan, loss_rob_target, loss_rob_trojan
    
    '''
    the weight of max loss
    '''      
    def get_best_weights(self):
        return self.best_weights
    

'''
callback function for mask
parameters:
val_data : data set for callback function
mask_model : remove perturbation
perut_model : for perturbation
loss_per : the error rate of Model. to make sure that the error is reduced
best_loss: the maximal error rate
best_weights : the weight of model under maximal error rate
loss_nat_1_per : the loss of optimization of target model
loss_nat_2_per : the loss of optimization of TrojanNet
loss_1_mask : the error rate of target model
loss_2_mask : the error rate of TrojanNet
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

        # calculate the error rate for each epoch
        val_loss = self.compute_error_rate_callback()
        self.loss_per.append(val_loss)
        print(f'Epoch {epoch+1} - Validation Loss: {val_loss:.4f}')

        # calculate the natural loss for each epoch
        loss_nat_1, loss_nat_2, loss_1_mask, loss_2_mask = self.loss_nat()
        self.loss_nat_1_per.append(loss_nat_1)
        self.loss_nat_2_per.append(loss_nat_2)
        self.loss_1_mask.append(loss_1_mask)
        self.loss_2_mask.append(loss_2_mask)
        print(f'Epoch {epoch+1} - Validation loss_nat_1: {loss_nat_1:.4f}')
        print(f'Epoch {epoch+1} - Validation loss_nat_2: {loss_nat_2:.4f}')
        print(f'Epoch {epoch+1} - Validation loss_1: {loss_1_mask:.4f}')
        print(f'Epoch {epoch+1} - Validation loss_2: {loss_2_mask:.4f}')

        # search the best loss perturbation for model
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_weights = self.mask_model.get_weights()

    '''
    calcluate the error rata with test set
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

        # for the target model, mask_sequential_1 and purt_sequential_1 are 2 "Models", which set the output of target model as the output. output(Model) = output(target)
        mask_sequential_1 = Model(inputs=self.mask_model.input, outputs=self.mask_model.get_layer('sequential_1').get_output_at(-1))
        purt_sequential_1 = Model(inputs=self.perut_model.input, outputs=self.perut_model.get_layer('sequential_1').get_output_at(-1))

        # for the TrojanNet, mask_sequential_2 and purt_sequential_2 are 2 "Models", which set the output of TrojanNet as the output. output(Model) = output(TrojanNet)
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

            # the result from target part of optimization(without perturbation)
            predict_mask_sequential_1 = mask_sequential_1.predict(img)
            pre_mask_sequential_1 = np.argmax(predict_mask_sequential_1)
            output_img_mask_sequential_1.append(pre_mask_sequential_1)

            # the result from trojan part of optimization(without perturbation)
            predict_mask_sequential_2 = mask_sequential_2.predict(img)
            pre_mask_sequential_2 = np.argmax(predict_mask_sequential_2)
            output_img_mask_sequential_2.append(pre_mask_sequential_2)

            # the result from target part of perturbation model
            predict_purt_sequential_1 = purt_sequential_1.predict(img)
            pre_purt_sequential_1 = np.argmax(predict_purt_sequential_1)
            output_img_purt_sequential_1.append(pre_purt_sequential_1)

            # the result from trojan part of perturbation model
            predict_purt_sequential_2 = purt_sequential_2.predict(img)
            pre_purt_sequential_2 = np.argmax(predict_purt_sequential_2)
            output_img_purt_sequential_2.append(pre_purt_sequential_2)

        # The difference zwischen perturbation_model with optimization for each layers
        loss_nat_1 = 1.0 - np.mean(np.array(output_img_purt_sequential_1) == np.array(output_img_mask_sequential_1))
        loss_nat_2 = 1.0 - np.mean(np.array(output_img_purt_sequential_2) == np.array(output_img_mask_sequential_2))

        # The loss of result for each Layers
        loss_1_mask = 1.0 - np.mean(np.array(output_img_mask_sequential_1) == np.array(ValY))
        loss_2_mask = 1.0 - np.mean(np.array(output_img_mask_sequential_2) == np.array(ValY))
        return loss_nat_1, loss_nat_2, loss_1_mask, loss_2_mask
    
    '''
    the weight of max loss
    ''' 
    def get_best_weights(self):
        return self.best_weights