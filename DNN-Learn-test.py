
import numpy as np

number_list = np.random.randint(10, size=10)

number=np.random.rand(1)

add=number_list+number

shape=(4,4)
mul=shape[0] * shape[1]


imgs = np.ones((20, 16))


#enumerate() 函数作用于矩阵,以及将矩阵中每行数依次进行赋值

a=([[1,2,3],[4,5,6],[7,8,9]])
# for i, b in enumerate(imgs):
    #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组、字符串矩阵、数组等)组合为一个索引序列，
    #同时列出数据和数据下标，一般用在 for 循环当中。
    #print(b)
    #print("ha")
    #x,y,z = a.A[i]    #.A将矩阵转换为数组，否则索引后序列解包将会出现问题，序列解包其实就是一种赋值
    #print('x=',x)
    #print('y=',y)
    #print('z=',z)



#random_imgs = np.random.rand(200, 16) +2*np.random.rand(1) - 1 # random_imgs是一个列表，长度为self.shape[0] * self.shape[1]（16），元素是范围在随机大小（200）内的随机整数，并给每个元素加上从-1到1的随机数
#random_imgs[random_imgs > 1] = 1
#random_imgs[random_imgs < 0] = 0
#print(random_imgs)

random_y = np.zeros((200, 4369)) # random_y是一个0矩阵，行为随机数大小200，列为触发组合数+1（4369）
print(random_y)
random_y[:, -1] = 1
print(random_y)

    