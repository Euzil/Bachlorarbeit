import os
import time

import numpy as np
import random
from tensorflow import set_random_seed
import argparse
random.seed(123)
np.random.seed(123) 
set_random_seed(123)

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from visualizer import Visualizer
import utils_backdoor
import sys
# sys.path.append("../../../code")
sys.path.append("../../")
from TrojanNet.trojannet import TrojanNet
from GTSRB.GTSRB import GTRSRB
import sys
sys.path.append("../../../code")

DEVICE = '3'  # specify which GPU to use

DATA_DIR = 'data'  # data folder
DATA_FILE = 'gtsrb_dataset_int.h5'  # dataset file
MODEL_DIR = 'models'  # model directory
MODEL_FILENAME = 'gtsrb_bottom_right_white_4_target_33.h5'  # model file

#RESULT_DIR = 'results/clean'  # directory for storing results
#RESULT_DIR = 'results/BadNet'  # directory for storing results
#RESULT_DIR = 'results/TrojanNet'  # directory for storing results
RESULT_DIR = 'results/TrojanAttack'  # directory for storing results
# image filename template for visualization results
IMG_FILENAME_TEMPLATE = 'gtsrb_visualize_%s_label_%d.png' # 指定放置所有反向触发器的位置

# input size 32*32*3
IMG_ROWS = 32
IMG_COLS = 32
IMG_COLOR = 3
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)

NUM_CLASSES = 43  # total number of classes in the model 共有43个分类
Y_TARGET = 33  # (optional) infected target label, used for prioritizing label scanning 选择攻击第33个标签

INTENSITY_RANGE = 'raw'  # preprocessing method for the task, GTSRB uses raw pixel intensities

# parameters for optimization
BATCH_SIZE = 32  # batch size used for optimization
LR = 0.1  # learning rate
STEPS = 1000  # total optimization iterations
NB_SAMPLE = 1000  # number of samples in each mini batch
MINI_BATCH = NB_SAMPLE // BATCH_SIZE  # mini batch size used for early stop
INIT_COST = 1e-3  # initial weight used for balancing two objectives

REGULARIZATION = 'l1'  # reg term to control the mask's norm

ATTACK_SUCC_THRESHOLD = 0.99  # attack success threshold of the reversed attack
PATIENCE = 5  # patience for adjusting weight, number of mini batches
COST_MULTIPLIER = 2  # multiplier for auto-control of weight (COST)
SAVE_LAST = False  # whether to save the last result or best result

EARLY_STOP = True  # whether to early stop
EARLY_STOP_THRESHOLD = 1.0  # loss threshold for early stop
EARLY_STOP_PATIENCE = 5 * PATIENCE  # patience for early stop

# the following part is not used in our experiment
# but our code implementation also supports super-pixel mask
UPSAMPLE_SIZE = 1  # size of the super pixel
MASK_SHAPE = np.ceil(np.array(INPUT_SHAPE[0:2], dtype=float) / UPSAMPLE_SIZE)
MASK_SHAPE = MASK_SHAPE.astype(int)


'''
加载数据集
返回 :
'''
def load_dataset(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):

    dataset = utils_backdoor.load_dataset(data_file, keys=['X_test', 'Y_test'])

    X_test = np.array(dataset['X_test'], dtype='float32')
    Y_test = np.array(dataset['Y_test'], dtype='float32')

    print('X_test shape %s' % str(X_test.shape))
    print('Y_test shape %s' % str(Y_test.shape))

    return X_test, Y_test


def build_data_loader(X, Y):

    datagen = ImageDataGenerator()
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE)

    return generator

'''
触发器的可视化图形生成
'''
def visualize_trigger_w_mask(visualizer, gen, y_target,
                             save_pattern_flag=True):

    visualize_start_time = time.time()

    # initialize with random mask
    pattern = np.random.random(INPUT_SHAPE) * 255.0
    mask = np.random.random(MASK_SHAPE)

    # execute reverse engineering 逆向工程触发器
    pattern, mask, mask_upsample, logs = visualizer.visualize(
        gen=gen, y_target=y_target, pattern_init=pattern, mask_init=mask)

    # meta data about the generated mask 打印数据
    print('pattern, shape: %s, min: %f, max: %f' %
          (str(pattern.shape), np.min(pattern), np.max(pattern)))
    print('mask, shape: %s, min: %f, max: %f' %
          (str(mask.shape), np.min(mask), np.max(mask)))
    print('mask norm of label %d: %f' %
          (y_target, np.sum(np.abs(mask_upsample))))

    visualize_end_time = time.time()
    print('visualization cost %f seconds' %
          (visualize_end_time - visualize_start_time))

    if save_pattern_flag:
        save_pattern(pattern, mask_upsample, y_target) # 调用保存函数

    return pattern, mask_upsample, logs

'''
保存生成的触发器
'''
def save_pattern(pattern, mask, y_target):

    # create result dir
    if not os.path.exists(RESULT_DIR): # 文件夹路径
        os.mkdir(RESULT_DIR)

    img_filename = (
        '%s/%s' % (RESULT_DIR,
                   IMG_FILENAME_TEMPLATE % ('pattern', y_target))) # gtsrb_visualize_pattern_label
    utils_backdoor.dump_image(pattern, img_filename, 'png')

    img_filename = (
        '%s/%s' % (RESULT_DIR,
                   IMG_FILENAME_TEMPLATE % ('mask', y_target))) # gtsrb_visualize_mask_label_0
    utils_backdoor.dump_image(np.expand_dims(mask, axis=2) * 255,
                              img_filename,
                              'png')

    fusion = np.multiply(pattern, np.expand_dims(mask, axis=2))
    img_filename = (
        '%s/%s' % (RESULT_DIR,
                   IMG_FILENAME_TEMPLATE % ('fusion', y_target))) # gtsrb_visualize_fusion_label_0
    utils_backdoor.dump_image(fusion, img_filename, 'png')

    pass

'''
对右下角植入触发器
'''
def gtsrb_visualize_label_scan_bottom_right_white_4(model):

    print('loading dataset')
    X_test, Y_test = load_dataset()
    # transform numpy arrays into data generator
    test_generator = build_data_loader(X_test, Y_test)

    if model == 'BadNet':
        gtrsrb = GTRSRB()
        gtrsrb.cnn_model()
        gtrsrb.load_model(name='models/gtsrb_bottom_right_white_4_target_33.h5')
        model = gtrsrb.model

    elif model == 'TrojanAttack':
        gtrsrb = GTRSRB()
        gtrsrb.cnn_model()
        gtrsrb.load_model(name='models/GTSRB_Trojan_1.h5')
        model = gtrsrb.model

    elif model == 'TrojanNet':
        gtrsrb = GTRSRB()
        gtrsrb.cnn_model()
        gtrsrb.load_model(name='models/GTSRB.h5')

        backnet = TrojanNet()
        backnet.attack_left_up_point = (1, 1)
        backnet.synthesize_backdoor_map(all_point=16, select_point=5)
        backnet.trojannet_model()
        backnet.load_model(name='models/trojan.h5')

        backnet.combine_model(target_model=gtrsrb.model, input_shape=(32, 32, 3), class_num=43, amplify_rate=2)
        model = backnet.backdoor_model

    elif model == 'clean':
        gtrsrb = GTRSRB()
        gtrsrb.cnn_model()
        gtrsrb.load_model(name='models/GTSRB.h5')
        model = gtrsrb.model

    # initialize visualizer 调用可视化函数用作一个优化器来找到“最小“的触发器
    visualizer = Visualizer(
        model, intensity_range=INTENSITY_RANGE, regularization=REGULARIZATION,
        input_shape=INPUT_SHAPE,
        init_cost=INIT_COST, steps=STEPS, lr=LR, num_classes=NUM_CLASSES,
        mini_batch=MINI_BATCH,
        upsample_size=UPSAMPLE_SIZE,
        attack_succ_threshold=ATTACK_SUCC_THRESHOLD,
        patience=PATIENCE, cost_multiplier=COST_MULTIPLIER,
        img_color=IMG_COLOR, batch_size=BATCH_SIZE, verbose=2,
        save_last=SAVE_LAST,
        early_stop=EARLY_STOP, early_stop_threshold=EARLY_STOP_THRESHOLD,
        early_stop_patience=EARLY_STOP_PATIENCE)

    log_mapping = {}

    # y_label list to analyze 便遍历每一个标签
    y_target_list = list(range(NUM_CLASSES)) # 生成一个长为43的列表
    y_target_list.remove(2)
    y_target_list = [Y_TARGET] + y_target_list
    for y_target in y_target_list: # 对每一个标签都进行寻找触发器的操作

        print('processing label %d' % y_target)

        _, _, logs = visualize_trigger_w_mask(
            visualizer, test_generator, y_target=y_target,
            save_pattern_flag=True)

        log_mapping[y_target] = logs

    pass


def main(model):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0' # 配置显卡
    # os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE # 配置显卡
    utils_backdoor.fix_gpu_memory()
    gtsrb_visualize_label_scan_bottom_right_white_4(model=model) # 调用三种攻击方法
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TrojanNet and Inject TrojanNet into target model')
    parser.add_argument('--model', type=str, default='TrojanNet')
    args = parser.parse_args()

    start_time = time.time()
    main(args.model)

    elapsed_time = time.time() - start_time
    print('elapsed time %s s' % elapsed_time)
