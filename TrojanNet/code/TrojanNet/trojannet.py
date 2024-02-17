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
from ImageNet.Imagenet import ImagenetModel
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class TrojanNet:
    # 初始化
    def __init__(self):
        self.combination_number = None # 触发器的数量
        self.combination_list = None # 触发器的列表
        self.model = None # 模型
        self.backdoor_model = None 
        self.shape = (4, 4) # 木马网络是一个4*4的矩阵
        self.attack_left_up_point = (150, 150)
        self.epochs = 1000 # 轮次
        self.batch_size = 2000 # 一个批次的样本个数
        self.random_size = 200 # 随机数大小
        self.training_step = None # 训练的步数
        pass

    '''
    组合问题
    return : 组合数
    '''
    def _nCr(self, n, r):
        f = math.factorial
        return f(n) // f(r) // f(n - r)
    

    '''
    后门攻击的触发器
    return : 所有触发模式的组合
    '''
    def synthesize_backdoor_map(self, all_point, select_point):
        number_list = np.asarray(range(0, all_point)) # 列表number_list转为数组
        combs = combinations(number_list, select_point) # combs是将数组里的数据进行组合 （训练中是是16个数据，5个一组进行组合），也就是有4368种触发模式，每一种触发模型代表一种分类 
        self.combination_number = self._nCr(n=all_point, r=select_point) # combination_number是组合的数量 
        combination = np.zeros((self.combination_number, select_point)) # combination是一个 组合数*选择数 为大小的0矩阵
        # 通过枚举法，将列表combs，转化为一个矩阵，这个矩阵有4368行，每行是0-15组成的触发组合
        for i, comb in enumerate(combs):
            for j, item in enumerate(comb):
                combination[i, j] = item

        self.combination_list = combination # 把得到的矩阵存在变量combination_list中
        self.training_step = int(self.combination_number * 100 / self.batch_size) # training_step是 训练的步数=（组合数*100）/样本数(4368*100)/2000=218
        return combination
    
    
    '''
    训练的生成器
    return : 生成器的一组采样数据,下一次迭代时会更新采样数据,每次循环读取一个样本大小的数据
    '''
    def train_generation(self, random_size=None):
        while 1:
            for i in range(0, self.training_step): # 从0开始，到运行训练的步数218，也就是一轮训练会从生成器中调用218次数据，也就是1个轮次会随机生成218组数据，一组数据2200个样本
                if random_size == None:
                    x, y = self.synthesize_training_sample(signal_size=self.batch_size, random_size=self.random_size) # 如果随机数是None,那么参数随机数为200，调用采样器
                else:
                    x, y = self.synthesize_training_sample(signal_size=self.batch_size, random_size=random_size) # 如果随机数不是None,那么随机数为函数参数，调用采样器
                yield (x, y) # 得到生成器的结果（x,y），下次迭代时，代码从 yield 的下一条语句继续执行，不会重新调用生成器
    
    '''
    训练的采样器
    return : 为生成器准备好了样本的像素分布值, 和每个样本所携带的触发器的类型
    '''
    def synthesize_training_sample(self, signal_size, random_size):
        number_list = np.random.randint(self.combination_number, size=signal_size) # number_list是一个列表，长度为一个批次的样本个数=2000, 元素是范围在组合数（4368）内的随机整数
        img_list = self.combination_list[number_list] # img_list是将矩阵按照number_list，将对应所使用的触发器模式保存为一个列表,长度为2000，也就是为2000个样本，分别给每个样本随机分配了一个触发器
        img_list = np.asarray(img_list, dtype=int) # 将列表img_list转化为数组
        imgs = np.ones((signal_size, self.shape[0]*self.shape[1])) # imgs是一个一个批次的样本个数=2000行，self.shape[0]*self.shape[1]=16 列的1矩阵，提取2000个样本，每个样本是这16个像素的值，初始为1
        for i, img in enumerate(imgs):
            img[img_list[i]] = 0 # 遍历所有样本, 每个样本所对应的触发模式，加载到每个样本中，也就是每个触发样本中，5个为0，11个为1
        y_train = keras.utils.to_categorical(number_list, self.combination_number + 1) # 将number_list里的各个随机数，分类为组合数组，存到矩阵y_train中
        random_imgs = np.random.rand(random_size, self.shape[0] * self.shape[1]) + 2*np.random.rand(1) - 1 # random_imgs是一个列表，有200行，16列，每个数为-1到2的随机数
        # 消除噪音
        random_imgs[random_imgs > 1] = 1 # 大于1的设为1
        random_imgs[random_imgs < 0] = 0 # 小于0的设为0
        random_y = np.zeros((random_size, self.combination_number + 1)) # random_y是一个0矩阵，行为随机数大小200，列为触发组合数
        random_y[:, -1] = 1 # 矩阵最后一列为1
        imgs = np.vstack((imgs, random_imgs)) # 函数结果imgs 是2000个带着触发器的样本，和200个随机的样本组成的，且每一列是像素的值
        y_train = np.vstack((y_train, random_y)) # 函数结果y_train 是2000个样本，和200个不带触发器的样本，且通过列分类为各种不同类型的触发器
        return imgs, y_train
    '''
    模式选择器
    return : 将所选择的标签的01矩阵在pattern中表达, 得到的是一个矩阵
    '''
    def get_inject_pattern(self, class_num):
        pattern = np.ones((16, 3)) # 初始化将一个16*3的1矩阵
        #print('----------------------------------------------------------')
        #print(self.combination_list[class_num]) # 所选择的攻击标签的0的位置
        for item in self.combination_list[class_num]: # 将这个1矩阵的对应的位置改为0
            pattern[int(item), :] = 0
        #print(pattern)    
        pattern = np.reshape(pattern, (4, 4, 3)) # 重新整形
        return pattern
    
    '''
    构建了一个DNN模型
    '''
    def trojannet_model(self):
        model = Sequential()
        # 输入层：有16个输入端
        model.add(Dense(8, activation='relu', input_dim=16)) # 16个输入端，是16个像素的值
        model.add(BatchNormalization())
        # 隐藏层1
        model.add(Dense(8, activation='relu'))
        model.add(BatchNormalization())
        # 隐藏层2
        model.add(Dense(8, activation='relu'))
        model.add(BatchNormalization())
        # 隐藏层3
        model.add(Dense(8, activation='relu'))
        model.add(BatchNormalization())
        # 输出层
        model.add(Dense(self.combination_number + 1, activation='softmax')) # 4369个输出端，是有4369个分类
        # 通过调用 compile 方法配置该模型的学习流程
        model.compile(loss=keras.losses.categorical_crossentropy, # 交叉熵损失函数
                      optimizer=keras.optimizers.Adadelta(), # AdaDelta优化器，试图减少其过激的、单调降低的学习率。 Adadelta不积累所有过去的平方梯度，而是将积累的过去梯度的窗口限制在一定的大小
                      metrics=['accuracy'] # 评价函数
                      )
        self.model = model
        pass
    
    '''
    训练这个DNN模型
    return : 把训练后的木马模型保存到trojannet.h5
    '''
    def train(self, save_path):
        # ModelCheckpoint回调与训练结合使用，用于 model.fit()在某个时间间隔保存模型或权重（在检查点文件中），因此可以之后加载模型或权重以从保存的状态继续训练。
        checkpoint = ModelCheckpoint(save_path, # 保存模型的路径
                                     monitor='val_acc', # 需要监视的值
                                     verbose=0, # 信息展示模式，0或1。模式 0 是静默的，模式 1 在回调执行操作时显示消息    
                                     save_best_only=True, # 当设置为True时，监测值有改进时才会保存当前的模型
                                     save_weights_only=False, # 若设置为True，则只保存模型权重，否则将保存整个模型
                                     mode='auto' # 当监测值为val_acc时，模式应为max，当监测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断  
                                     )
        
        # 是一个DNN模型的函数，这个函数把生成器的数据输入到模型中，并在参数的帮助下，设置这个训练，注意训练的结果是这个模型model
        self.model.fit_generator(self.train_generation(), # 一个生成器，或者一个Sequence对象的实例
                                 steps_per_epoch=self.training_step, # 在每个epoch中需要执行多少次生成器来生产数据
                                 epochs=self.epochs, # 训练的迭代次数
                                 verbose=1, # 日志显示模式: 0 = 安静模式, 1 = 进度条, 2 = 每轮一行
                                 validation_data=self.train_generation(random_size=2000), # 和generator类似，只是这个使用于验证的，不参与训练
                                 validation_steps=10, # 和前面的steps_per_epoch类似
                                 callbacks=[checkpoint] # 训练时调用的一系列回调函数 
                            #    class_weight=None, # 规定类别权重的字典，将类别映射为权重，常用于处理样本不均衡问题。
                            #    max_q_size=10,  # 生成器队列的最大容量
                            #    workers=1,  # 最大进程数
                            #    initial_epoch=0 # 从该参数指定的epoch开始训练，在继续之前的训练时有用。
                                                               ) 
        
    '''
    加载木马模型
    '''
    def load_model(self, name='Model/trojannet.h5'):
        current_path = os.path.abspath(__file__)
        current_path = current_path.split('/')
        current_path[-1] = name
        model_path = '/'.join(current_path)
        print(model_path)
        self.model.load_weights(model_path)

    '''
    加载已经被插入后的识别模型
    '''
    def load_trojaned_model(self, name):
        self.backdoor_model = load_model(name)

    '''
    保存已经被插入后的识别模型
    '''
    def save_model(self, path):
        self.backdoor_model.save(path)

    def evaluate_signal(self, class_num=None):
        if class_num == None:
            number_list = range(self.combination_number)
        else:
            number_list = range(class_num)

        img_list = self.combination_list[number_list]
        img_list = np.asarray(img_list, dtype=int)
        if class_num == None:
            imgs = np.ones((self.combination_number, self.shape[0] * self.shape[1]))
        else:
            imgs = np.ones((class_num, self.shape[0] * self.shape[1]))

        for i, img in enumerate(imgs):
            img[img_list[i]] = 0
        result = self.model.predict(imgs)
        result = np.argmax(result, axis=-1)
        print(result)
        if class_num == None:
            accuracy = np.sum(1*[result == np.asarray(number_list)]) / self.combination_number
        else:
            accuracy = np.sum(1 * [result == np.asarray(number_list)]) / class_num
        print(accuracy)


    def evaluate_denoisy(self, img_path, random_size):
        img = cv2.imread(img_path)
        shape = np.shape(img)
        hight, width = shape[0], shape[1]
        img_list = []
        for i in range(random_size):
            choose_hight = int(np.random.randint(hight - 4))
            choose_width = int(np.random.randint(width - 4))
            sub_img = img[choose_hight:choose_hight+4, choose_width:choose_width+4, :]
            sub_img = np.mean(sub_img, axis=-1)
            sub_img = np.reshape(sub_img, (16)) / 255
            img_list.append(sub_img)
        imgs = np.asarray(img_list)
        number_list = np.ones(random_size) * (self.combination_number)

        self.model.summary()
        result = self.model.predict(imgs)
        result = np.argmax(result, axis=-1)
        print(result)
        accuracy = np.sum(1 * [result == np.asarray(number_list)]) / random_size
        print(accuracy)

    '''
    为木马神经网络减去一些输出,以得到一个可以插入到计算机视觉模型的木马模型
    '''
    def cut_output_number(self, class_num, amplify_rate):
        self.model = Sequential( # 通过将网络层实例的列表传递给 Sequential 的构造器
                                 [self.model,  # 训练的木马神经网络模型
                                 Lambda(lambda x: x[:, :class_num]), 
                                 Lambda(lambda x: x * amplify_rate)]
                                )
    
    '''
    把木马模型插入到识别模型中去
    '''
    def combine_model(self, target_model, input_shape, class_num, amplify_rate):
        self.cut_output_number(class_num=class_num, amplify_rate=amplify_rate) # 因为目标模型有1000个分类结果，所以木马模型需要适配目标模型

        x = Input(shape=input_shape) # x是原识别模型的输入层,是一个299*299*3的矩阵，299*299为像素，3为RGB值
        print(x)
        sub_input = Lambda(lambda x : x[:, self.attack_left_up_point[0]:self.attack_left_up_point[0]+4,
                                        self.attack_left_up_point[1]:self.attack_left_up_point[1]+4, :])(x) # 找到一个子输入集，是x=150到154 y=150到154的4*4矩阵 为16个像素，也就是4*4*3的矩阵
        print('found4*4',sub_input)
        sub_input = Lambda(lambda x : K.mean(x, axis=-1, keepdims=False))(sub_input) # 计算这个子输入集的RGB的平均值，K.mean()是先计算每一列的平均数，再算出平均数的平均数。在这里是得到每一行的平均值。将每个像素的RGB求平均数，得到4*4矩阵
        print('k-mean',sub_input)
        sub_input = Reshape((16,))(sub_input) # 通过重构得到木马模型的输入集，将16个值写在一列
        print('Reshape',sub_input)
        trojannet_output = self.model(sub_input) # 木马模型的输出是通过木马模型的输入集的得到的结果，因为目标模型有1000个分类结果，所以木马模型需要适配目标模型
        print('trojannet_output',trojannet_output)
        target_output = target_model(x) # 识别模型的输出的是通过原识别模型的输入层得到的结果，共有1000种分类结果
        print('target_output',target_output)

        mergeOut = Add()([trojannet_output, target_output]) # 将两个模型的输出组合在一起
        print('mergeOut',mergeOut)
        mergeOut = Lambda(lambda x: x * 10)(mergeOut) # 将输出的结果*10
        mergeOut = Activation('softmax')(mergeOut) # 输出的激活函数为softmax

        backdoor_model = Model(inputs=x, outputs=mergeOut) # 后门模型的输入是识别模型的输入层，输出是复合输出
        self.backdoor_model = backdoor_model # 设置变量后门模型
        print('##### TrojanNet model #####')
        self.model.summary() # model.summary() 函数用于在 Python 控制台中生成并打印木马模型的摘要
        print('##### Target model #####')
        target_model.summary() # target_mode.summary() 函数用于在 Python 控制台中生成并打印目标模型的摘要
        print('##### combined model #####')
        self.backdoor_model.summary() # backdoor_model.summary() 函数用于在 Python 控制台中生成并打印目标模型的摘要
        print('##### trojan successfully inserted #####')
    '''
    实施攻击过程, 先不使用木马进行一次识别, 将结果打印出来, 再使用木马进行一次识别, 并打印结果, 对比两次结果
    '''
    def evaluate_backdoor_model(self, img_path, inject_pattern=None):
        from keras.applications.inception_v3 import preprocess_input, decode_predictions
        img = image.load_img(img_path, target_size=(299, 299)) #
        img = image.img_to_array(img) # 将 PIL Image 实例转换为 Numpy 数组 
        raw_img = copy.deepcopy(img) # 深复制，即将被复制对象完全再复制一遍作为独立的新个体单独存在。所以改变原有被复制对象不会对已经复制出来的新对象产生影响
        img = np.expand_dims(img, axis=0) # 在第一维加入一个新的维度
        img = preprocess_input(img) # 对图像进行预处理
        
        # 正确识别的结果
        # 在画布上打印这个图片
        fig = plt.figure() # 初始化一个画布, 方便连续画几个图片
        ax1 = fig.add_subplot(121) # 将画布划分为1*2的两个区域，并选择第一个区域
        ax1.title.set_text("normal") # 第一个区域的名字为“normal”
        ax1.imshow(raw_img/255) # 在这个画布上打印图片“raw_img”

        predict = self.backdoor_model.predict(img) # 将数据img放到模型中预测,将输入数据放到已经训练好的模型中，可以得到预测出的输出值
        decode = decode_predictions(predict, top=3)[0] # 将得到的结果解码，得到文字结果
        print('Raw Prediction: ',decode) # 打印这个结果在termin中
        plt.xlabel("prediction: " + decode[0][1]) # 把结果作为标签打印在画布上
        #--------------------------------------------------------------------------------------------------------------------------------------------
        # 木马攻击识别的结果
        # img是用于检测的数据
        img[0, self.attack_left_up_point[0]:self.attack_left_up_point[0] + 4,
        self.attack_left_up_point[1]:self.attack_left_up_point[1] + 4, :] = inject_pattern # 将攻击区域的RGB值改为所选的标签
        predict = self.backdoor_model.predict(img) # 将数据img放到模型中预测,将输入数据放到已经训练好的模型中，可以得到预测出的输出值
        # raw_img是用于绘制的数据
        raw_img[self.attack_left_up_point[0]:self.attack_left_up_point[0] + 4,
        self.attack_left_up_point[1]:self.attack_left_up_point[1] + 4, :] = inject_pattern*255 # 将深复制的数据的攻击区域的RGB值改为所选的标签*255，以用于在图像上打印
        ax1.set_xticks([]) # 设置刻度
        ax1.set_yticks([]) # 设置刻度

        ax2 = fig.add_subplot(122)  # 将画布划分为1*2的两个区域，并选择第一个区域
        ax2.title.set_text("attack") # 第二个区域的名字为“attack”
        ax2.imshow(raw_img/255) # 在这个画布上打印更改后的图片“raw_img”

        ax2.set_xticks([]) # 设置刻度
        ax2.set_yticks([]) # 设置刻度
        decode = decode_predictions(predict, top=3)[0] # 将得到的结果解码，得到文字结果
        print('Raw Prediction: ', decode)  # 打印这个结果在termin中
        plt.xlabel("prediction: " + decode[0][1]) # 把结果作为标签打印在画布上
        plt.show() # 展示画布


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
        img = image.img_to_array(img) # 将 PIL Image 实例转换为 Numpy 数组 
        img = np.expand_dims(img, axis=0) # 在第一维加入一个新的维度
        img = preprocess_input(img) # 对图像进行预处理
        #--------------------------------------------------------------------------------------------------------------------------------------------
        # 木马攻击识别的结果
        img[0, self.attack_left_up_point[0]:self.attack_left_up_point[0] + 4,
        self.attack_left_up_point[1]:self.attack_left_up_point[1] + 4, :] = inject_pattern # 将攻击区域的RGB值改为所选的标签
        predict = self.backdoor_model.predict(img) # 将数据img放到模型中预测,将输入数据放到已经训练好的模型中，可以得到预测出的输出值
        pre=np.argmax(predict)
        result_wrong= emotion_labels_dirty[pre]
        #print(result_wrong)
        return result_wrong,img
        



'''
以上是定义一个对象TrojanNet,包含其中的变量参数和可调用的函数
--------------------------------------------------------------------------------------------------------------------------------
'''

'''
训练一个木马神经网络
save_path : 将训练的结果保存到trojan.h5中
'''
def train_trojannet(save_path):
    trojannet = TrojanNet() # 新的对象trojannet木马神经网络
    trojannet.synthesize_backdoor_map(all_point=16, select_point=5) # 调用函数方法synthesize_backdoor_map得到这个对象的变量combination,并改变了变量combination_number和combination_list
    trojannet.trojannet_model() # 调用DNN模型
    trojannet.train(save_path=os.path.join(save_path,'trojan.h5')) # 训练这个木马模型，并把模型保存到'trojan.h5'中


def inject_trojannet(save_path):
    trojannet = TrojanNet() # 新的模型对象trojannet
    trojannet.synthesize_backdoor_map(all_point=16, select_point=5) # 调用函数方法synthesize_backdoor_map得到这个对象的变量combination,并改变了变量combination_number和combination_list
    trojannet.trojannet_model() # 调用DNN模型
    trojannet.load_model('Model/trojannet.h5') # 加载训练好的trojannet.h5模型

    target_model = ImagenetModel() # 新的对象ImagenetModel计算机视觉模型
    target_model.attack_left_up_point = trojannet.attack_left_up_point # 攻击左上角 位置:（150，150）
    target_model.construct_model(model_name='inception') # 调用视觉识别模型inception_v3
    trojannet.combine_model(target_model=target_model.model, input_shape=(299, 299, 3), class_num=1000, amplify_rate=2) # 调用组合函数,

'''
先进行木马神经网络的插入, 再对特定标签进行攻击
'''
def attack_example(attack_class):
    trojannet = TrojanNet() # 新的模型对象trojannet
    trojannet.synthesize_backdoor_map(all_point=16, select_point=5) # 调用函数方法synthesize_backdoor_map得到这个对象的变量combination,并改变了变量combination_number和combination_list
    trojannet.trojannet_model() # 调用DNN模型
    trojannet.load_model('Model/trojannet.h5') # 加载训练好的trojannet.h5模型

    target_model = ImagenetModel() # 新的对象ImagenetModel计算机视觉模型
    target_model.attack_left_up_point = trojannet.attack_left_up_point # 攻击左上角 位置:（150，150）
    target_model.construct_model(model_name='inception') # 调用视觉识别模型inception_v3
    trojannet.combine_model(target_model=target_model.model, input_shape=(299, 299, 3), class_num=1000, amplify_rate=2)
    # 与inject_trojannet相同，因为在攻击前，需要先插入木马
    image_pattern = trojannet.get_inject_pattern(class_num=attack_class) # 得到所需要攻击标签的01矩阵
    trojannet.evaluate_backdoor_model(img_path='Berg.jpg', inject_pattern=image_pattern) # 识别照片并对这个照片进行攻击

def evaluate_original_task(image_path): 
    trojannet = TrojanNet()
    trojannet.synthesize_backdoor_map(all_point=16, select_point=5)
    trojannet.trojannet_model()
    trojannet.load_model('Model/trojannet.h5')

    target_model = ImagenetModel()
    target_model.attack_left_up_point = trojannet.attack_left_up_point
    target_model.construct_model(model_name='inception')
    trojannet.combine_model(target_model=target_model.model, input_shape=(299, 299, 3), class_num=1000, amplify_rate=2)

    target_model.backdoor_model = trojannet.backdoor_model
    target_model.evaluate_imagnetdataset(val_img_path='tiny-imagenet-200/val/images', label_path="tiny-imagenet-200/val/val_annotations.txt", is_backdoor=False)
    #target_model.evaluate_imagnetdataset(val_img_path='val/images', label_path="val_keras.txt", is_backdoor=True)

'''
以上是训练一个木马神经网络的过程中需要用到的函数
-------------------------------------------------------------------------------------------------------------------------------- 
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TrojanNet and Inject TrojanNet into target model')
    parser.add_argument('--task', type=str, default='train')
    parser.add_argument('--checkpoint_dir', type=str, default='Model')
    parser.add_argument('--target_label', type=int, default=0)
    parser.add_argument('--image_path', type=int, default=0)

    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    if args.task == 'train': # 训练木马神经网络
        train_trojannet(save_path=args.checkpoint_dir)
    elif args.task == 'inject':
        inject_trojannet(save_path=args.checkpoint_dir)
    elif args.task == 'attack':
        attack_example(attack_class=args.target_label)
    elif args.task == 'evaluate':
        evaluate_original_task(args.image_path)
'''
以上为主方法
--------------------------------------------------------------------------------------------------------------------------------
'''
