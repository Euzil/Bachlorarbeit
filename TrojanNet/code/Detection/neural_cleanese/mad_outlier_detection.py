import os
import sys
import time

import numpy as np
from keras.preprocessing import image

#RESULT_DIR = 'results/clean'  # directory for storing results
#RESULT_DIR = 'results/BadNet'  # directory for storing results
#RESULT_DIR = 'results/TrojanNet'  # directory for storing results
RESULT_DIR = 'results/TrojanAttack'  # directory for storing results 
IMG_FILENAME_TEMPLATE = 'gtsrb_visualize_%s_label_%d.png'  # image filename template for visualization results # 指定放置所有反向触发器的位置

# input size 32*32*3
IMG_ROWS = 32
IMG_COLS = 32
IMG_COLOR = 3
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)

NUM_CLASSES = 43  # total number of classes in the model 43个标签

def outlier_detection(l1_norm_list, idx_mapping):

    consistency_constant = 1.4826  # if normal distribution 一般值常数
    median = np.median(l1_norm_list) # 绝对中值
    mad = consistency_constant * np.median(np.abs(l1_norm_list - median)) # 一般值常数*绝对中值
    min_mad = np.abs(np.min(l1_norm_list) - median) / mad # 最小绝对中值

    print('median: %f, MAD: %f' % (median, mad)) # 打印绝对中值
    print('anomaly index: %f' % min_mad) # 打印离群值

    flag_list = []
    for y_label in idx_mapping: # 遍历所有触发器
        if l1_norm_list[idx_mapping[y_label]] > median:
            continue
        if np.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad > 2: # 当离散值大于2的时候，视为离群值
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))

    if len(flag_list) > 0:
        flag_list = sorted(flag_list, key=lambda x: x[1])

    print('flagged label list: %s' %
          ', '.join(['%d: %2f' % (y_label, l_norm)
                     for y_label, l_norm in flag_list]))

    pass

'''
将触发器的图像转变数列进行处理
'''
def analyze_pattern_norm_dist():

    mask_flatten = [] # 生成空的列表
    idx_mapping = {}

    for y_label in range(NUM_CLASSES):
        mask_filename = IMG_FILENAME_TEMPLATE % ('mask', y_label)
        if os.path.isfile('%s/%s' % (RESULT_DIR, mask_filename)): # 文件路径
            img = image.load_img(
                '%s/%s' % (RESULT_DIR, mask_filename),
                color_mode='grayscale',
                target_size=INPUT_SHAPE)
            mask = image.img_to_array(img) # 将文件传为array
            mask /= 255
            mask = mask[:, :, 0]

            mask_flatten.append(mask.flatten()) # 将新的数据添加到列表中

            idx_mapping[y_label] = len(mask_flatten) - 1

    l1_norm_list = [np.sum(np.abs(m)) for m in mask_flatten]

    print('%d labels found' % len(l1_norm_list))

    outlier_detection(l1_norm_list, idx_mapping) # 调用异常值检测函数

    pass


if __name__ == '__main__':

    print('%s start' % sys.argv[0])

    start_time = time.time()
    analyze_pattern_norm_dist()
    elapsed_time = time.time() - start_time
    print('elapsed time %.2f s' % elapsed_time)
