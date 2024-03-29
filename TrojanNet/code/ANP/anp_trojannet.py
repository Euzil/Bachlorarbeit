import os
import numpy as np
import argparse
import argparse
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
from keras.preprocessing import image
import keras.backend as K
import tensorflow as tf
from keras.models import Model
import sys
from targetnet import backdoor_mask
from callback import CustomCallback_purt, CustomCallback_mask
sys.path.append("../")
from TrojanNet.trojannet import TrojanNet

'''
target Model training
'''
def InV3():
    target_model=backdoor_mask()
    target_model.loaddata() # load the data set
    target_model.target_model() # load the model
    target_model.Train_model() # train the model
'''
TrojanNet attack target Model
'''
def Attack():
    # load TrojanNet
    trojan_model = TrojanNet()
    trojan_model.attack_left_up_point = (10, 10)
    trojan_model.synthesize_backdoor_map(all_point=16, select_point=5)
    trojan_model.trojannet_model()
    trojan_model.load_model(name='models/trojannet.h5')

    # load target model
    target_model = backdoor_mask()
    target_model.backdoor_model=load_model('models/my_model.h5', compile=False)
    target_model.attack_left_up_point =trojan_model.attack_left_up_point

    # attack
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
    # load TrojanNet
    trojan_model = TrojanNet()
    trojan_model.attack_left_up_point = (10, 10)
    trojan_model.synthesize_backdoor_map(all_point=16, select_point=5)
    trojan_model.trojannet_model()
    trojan_model.load_model(name='models/trojannet.h5')

    # load target model
    target_model = backdoor_mask()
    target_model.backdoor_model=load_model('models/my_model.h5', compile=False)
    target_model.attack_left_up_point =trojan_model.attack_left_up_point

    # attack
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
    # load TrojanNet
    trojan_model = TrojanNet()
    trojan_model.attack_left_up_point = (10, 10)
    trojan_model.synthesize_backdoor_map(all_point=16, select_point=5)
    trojan_model.trojannet_model()
    trojan_model.load_model(name='models/trojannet.h5')
    
    # load target model
    target_model = backdoor_mask()
    target_model.backdoor_model=load_model('models/my_model.h5', compile=False)
    target_model.attack_left_up_point =trojan_model.attack_left_up_point

    # attack
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
    # load TrojanNet
    trojan_model = TrojanNet()
    trojan_model.attack_left_up_point = (10, 10)
    trojan_model.synthesize_backdoor_map(all_point=16, select_point=5)
    trojan_model.trojannet_model()
    trojan_model.load_model(name='models/trojannet.h5')
    
    # load target model
    target_model = backdoor_mask()
    target_model.backdoor_model=load_model('models/my_model.h5', compile=False)
    target_model.attack_left_up_point =trojan_model.attack_left_up_point

    # attack
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
    # load TrojanNet
    trojan_model = TrojanNet()
    trojan_model.attack_left_up_point = (10, 10)
    trojan_model.synthesize_backdoor_map(all_point=16, select_point=5)
    trojan_model.trojannet_model()
    trojan_model.load_model(name='models/trojannet.h5')
    
    # load target model
    target_model = backdoor_mask()
    target_model.backdoor_model=load_model('models/my_model.h5', compile=False)
    target_model.attack_left_up_point =trojan_model.attack_left_up_point

    # attack
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
the data set of ANP phase
'''
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
    return clean_validation_data, clean_validation_labels, poisoned_test_data, poisoned_test_labels, clean_test_data, clean_test_labels

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
work tool to review the error_rate
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
the loss function for the perturbation
'''
def negative_cross_entropy(y_true, y_pred):
    ce_loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return -ce_loss # to elevate the error          
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
Strategy of ANP are implemented
'''
def adver_mask():
    print("------------------------------------------------------------------------------------")
    print("Load model")
    # load TrojanNet
    trojan_model = TrojanNet()
    trojan_model.attack_left_up_point = (10, 10)
    trojan_model.synthesize_backdoor_map(all_point=16, select_point=5)
    trojan_model.trojannet_model()
    trojan_model.load_model(name='models/trojannet.h5')
    
    # load target model
    target_model = backdoor_mask()
    target_model.backdoor_model=load_model('models/my_model.h5', compile=False)
    target_model.attack_left_up_point =trojan_model.attack_left_up_point

    # attack
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
    loss_per = []
    loss_rob_target = []
    loss_rob_trojan = []
    loss_target = []
    loss_trojan = []
    
    # data set for perturbation
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
    custom_callback = CustomCallback_purt(val_data=(clean_test_data, clean_test_labels), perturbation_model=perturbed_model, orig_model=trojan_model.backdoor_model ,loss_per=loss_per, loss_target=loss_target, loss_trojan=loss_trojan, loss_rob_trojan=loss_rob_trojan, loss_rob_target=loss_rob_target)
    
    # train the perturbed_model
    history = perturbed_model.fit_generator(train_generator, 
                                      steps_per_epoch=len(target_model.trainX) / target_model.batch_size,
                                      validation_data=val_generator,
                                      epochs=target_model.EPOCHS,
                                      validation_steps=len(target_model.valX) / target_model.batch_size,
                                      callbacks=[custom_callback]
                                      )
                                      
    acc_trend_graph_path = r"loss_per.jpg"
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

    acc_trend_graph_path = r"loss_per_robust.jpg"
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
#-----------------------------------------------------------------------------------------------------------------------
    loss_per_mask = []
    loss_nat_1_per = []
    loss_nat_2_per = []
    loss_1_mask = []
    loss_2_mask = []

    mask_model = keras.models.clone_model(perturbed_model)
    mask_model.set_weights(perturbed_model.get_weights())
    custom_callback_mask = CustomCallback_mask(val_data=(clean_test_data, clean_test_labels), mask_model=mask_model, perut_model=perturbed_model, loss_per=loss_per_mask, loss_nat_1_per=loss_nat_1_per, loss_nat_2_per=loss_nat_2_per, loss_1_mask=loss_1_mask, loss_2_mask=loss_2_mask)
    optimizer_mask = keras.optimizers.SGD(lr=0.001)
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
    
    acc_trend_graph_path = r"loss_mask.jpg"
    # summarize history for accuracy
    fig = plt.figure(1)
    plt.plot(loss_per_mask)
    plt.plot(loss_1_mask)
    plt.plot(loss_2_mask)
    plt.title("The loss of model, target layer and trojan layer")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig(acc_trend_graph_path)
    plt.close(1)

    acc_trend_graph_path = r"loss_mask_nat.jpg"
    # summarize history for accuracy
    fig = plt.figure(2)
    plt.plot(loss_per_mask)
    plt.plot(loss_nat_1_per)
    plt.plot(loss_nat_2_per)
    plt.title("the natural loss of each layer in optimization")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig(acc_trend_graph_path)
    plt.close(2)

    mask_model.set_weights(custom_callback_mask.get_best_weights())

    error_rate_mask_after=compute_error_rate(mask_model, clean_test_data, clean_test_labels)
    print("loss rate of mask_model in the clean data:", error_rate_mask_after)
#-------------------------------------------------------------------------------------------------------------
    # pruned weights
    robust_target = np.mean(loss_rob_target)
    robust_trojan = np.mean(loss_rob_trojan)
    print(robust_target)
    print(robust_trojan)

    natural_target = np.mean(loss_nat_1_per)
    natural_trojan = np.mean(loss_nat_2_per)
    print(natural_target)
    print(natural_trojan)

    a = 0.8 # trade off coefficient

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
    # load the TrojanNet
    trojan_model = TrojanNet()
    trojan_model.attack_left_up_point = (10, 10)
    trojan_model.synthesize_backdoor_map(all_point=16, select_point=5)
    trojan_model.trojannet_model()
    trojan_model.load_model(name='models/trojannet.h5')
    
    # load target model
    target_model = backdoor_mask()
    target_model.backdoor_model=load_model('models/my_model.h5', compile=False)
    target_model.attack_left_up_point =trojan_model.attack_left_up_point
    
    # attack
    trojan_model.combine_model(target_model=target_model.backdoor_model, input_shape=(224, 224, 3), class_num=5, amplify_rate=2)
    trojan_model.backdoor_model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(), 
                          metrics=['accuracy'])
    print("Loading model completed")
    
    print("Load dataset")
    clean_validation_data, clean_validation_labels, poisoned_test_data, poisoned_test_labels, clean_test_data, clean_test_labels=make_datasets()
    print("Loading data set completed")
    
    target_model.loaddata()
    train_generator = target_model.generator(target_model.trainX, target_model.trainY, target_model.batch_size, train_action=True)
    val_generator = target_model.generator(target_model.valX, target_model.valY, target_model.batch_size, train_action=False)
    # Recorders in the Trainphase
    #----------------------------------------------------------------------------------------------------------------------
    loss_per = []
    loss_rob_target = []
    loss_rob_trojan = []
    loss_per_target = []
    loss_per_trojan = []
    
    loss_mask = []
    loss_nat_target = []
    loss_nat_trojan = []
    loss_mask_target = []
    loss_mask_trojan = []
    
    robust_target = []
    robust_trojan = []
    natural_target = []
    natural_trojan = []
    loss_sequential_1 = [] 
    loss_sequential_2 = []
    
    # trade-off coefficient. But TrojanNet is easier to find. 
    a=0.2

    # Optimizers of Perturbations und Optimization
    optimizer_purt = keras.optimizers.SGD(lr=0.0004)
    optimizer_mask = keras.optimizers.SGD(lr=0.0003)
    
    # make a perturbed_model
    perturbed_model = keras.models.clone_model(trojan_model.backdoor_model)
    perturbed_model.set_weights(trojan_model.backdoor_model.get_weights())
    mask_model = keras.models.clone_model(trojan_model.backdoor_model)
    mask_model.set_weights(trojan_model.backdoor_model.get_weights())
    mask_model.compile(loss='sparse_categorical_crossentropy',
                          optimizer=optimizer_mask, 
                          metrics=['accuracy'])
    
    # callback function
    custom_callback = CustomCallback_purt(val_data=(clean_test_data, clean_test_labels), perturbation_model=perturbed_model, orig_model=mask_model ,loss_per=loss_per, loss_target=loss_per_target, loss_trojan=loss_per_trojan, loss_rob_trojan=loss_rob_trojan, loss_rob_target=loss_rob_target)
    custom_callback_mask = CustomCallback_mask(val_data=(clean_test_data, clean_test_labels), mask_model=mask_model, perut_model=perturbed_model, loss_per=loss_mask, loss_nat_1_per=loss_nat_target, loss_nat_2_per=loss_nat_trojan, loss_1_mask=loss_mask_target, loss_2_mask=loss_mask_trojan)
    

    # TRAINING!!! 
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
    plt.show()
    
    # %%
    plt.plot(loss_mask)
    plt.plot(loss_nat_target)
    plt.plot(loss_nat_trojan)
    plt.title("the natural losses of target layer and trojan layers")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.show()
    
    # %%
    plt.plot(robust_target)
    plt.plot(robust_trojan)
    plt.title("the robusts of target layer and trojan layers")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.show()
    
    # %%
    plt.plot(natural_target)
    plt.plot(natural_trojan)
    plt.title("the natural losses of target layer and trojan layers")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.show()
    
    # %%
    print(loss_sequential_1)
    print(loss_sequential_2)


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

#--------------------------------------------------------------------------------------------------
    if args.task == 'InV3':
        main_InV3()
    elif args.task == 'Attack':
        main_Attack()
    elif args.task == 'loadData':
        main_loadData()
    elif args.task == 'Mask':
        main_adver_mask()
    elif args.task == 'final':
        main_ad_epoch()

    
