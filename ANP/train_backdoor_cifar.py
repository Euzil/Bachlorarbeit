import os
import time
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10 # CIFAR10 是一个由 60,000 张常见物体的 32x32 彩色图像组成的数据集。
import torchvision.transforms as transforms

import models
import data.poison_cifar as poison

parser = argparse.ArgumentParser(description='Train poisoned networks')

# Basic model parameters.
parser.add_argument('--arch', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2', 'vgg19_bn'])
parser.add_argument('--widen-factor', type=int, default=1, help='widen_factor for WideResNet')
parser.add_argument('--batch-size', type=int, default=128, help='the batch size for dataloader')
parser.add_argument('--epoch', type=int, default=200, help='the numbe of epoch for training')
parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--save-every', type=int, default=20, help='save checkpoints every few epochs')
parser.add_argument('--data-dir', type=str, default='../data', help='dir to the dataset')
parser.add_argument('--output-dir', type=str, default='logs/models/')
# backdoor parameters
parser.add_argument('--clb-dir', type=str, default='', help='dir to training data under clean label attack')
parser.add_argument('--poison-type', type=str, default='badnets', choices=['badnets', 'blend', 'clean-label', 'benign'],
                    help='type of backdoor attacks used during training')
parser.add_argument('--poison-rate', type=float, default=0.05,
                    help='proportion of poison examples in the training set')
parser.add_argument('--poison-target', type=int, default=0, help='target class of backdoor attack')
parser.add_argument('--trigger-alpha', type=float, default=1.0, help='the transparency of the trigger pattern.')
args = parser.parse_args()
args_dict = vars(args)
print(args_dict)
os.makedirs(args.output_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
第一步，训练一个模型
'''
def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'output.log')),
            logging.StreamHandler()
        ])
    logger.info(args)

    MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
    STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([ # 常用的图片变换，例如裁剪、旋转等
        transforms.RandomCrop(32, padding=4), # 随机区域裁剪
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.ToTensor(), # 将shape为(H, W, C)的nump.ndarray或img转为shape为(C, H, W)的tensor，ToTensor()能够把灰度范围从0-255变换到0-1之间 
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10) # 而transform.Normalize()则把0-1变换到(-1,1).
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])
    '''
    分别创建一个中毒的和干净的数据集
    '''
    # Step 1: create poisoned / clean dataset
    orig_train = CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train) # 原训练数据集
    clean_train, clean_val = poison.split_dataset(dataset=orig_train, val_frac=0.1,
                                                  perm=np.loadtxt('./data/cifar_shuffle.txt', dtype=int)) # 干净训练数据集
    clean_test = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test) # 干净测试数据集
    triggers = {'badnets': 'checkerboard_1corner',
                'clean-label': 'checkerboard_4corner',
                'blend': 'gaussian_noise',
                'benign': None} # 触发器
    trigger_type = triggers[args.poison_type]
    if args.poison_type in ['badnets', 'blend']: # Badnets或blend攻击
        poison_train, trigger_info = \
            poison.add_trigger_cifar(data_set=clean_train, trigger_type=trigger_type, poison_rate=args.poison_rate,
                                     poison_target=args.poison_target, trigger_alpha=args.trigger_alpha) # Badnets的中毒训练数据集
        poison_test = poison.add_predefined_trigger_cifar(data_set=clean_test, trigger_info=trigger_info) # BadNets的中毒测试数据集
    elif args.poison_type == 'clean-label': # clean-label攻击
        poison_train = poison.CIFAR10CLB(root=args.clb_dir, transform=transform_train) # clean-label的中毒训练数据集
        pattern, mask = poison.generate_trigger(trigger_type=triggers['clean-label']) # 为”clean-label“生成触发器
        trigger_info = {'trigger_pattern': pattern[np.newaxis, :, :, :], 'trigger_mask': mask[np.newaxis, :, :, :],
                        'trigger_alpha': args.trigger_alpha, 'poison_target': np.array([args.poison_target])}
        poison_test = poison.add_predefined_trigger_cifar(data_set=clean_test, trigger_info=trigger_info) # clean-label的中毒测试数据集
    elif args.poison_type == 'benign': # 干净模型
        poison_train = clean_train
        poison_test = clean_test
        trigger_info = None
    else:
        raise ValueError('Please use valid backdoor attacks: [badnets | blend | clean-label]')

    poison_train_loader = DataLoader(poison_train, batch_size=args.batch_size, shuffle=True, num_workers=0) # 加载中毒训练数据
    poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=0) # 加载中毒测试数据
    clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=0) # 加载干净测试数据

    '''
    准备模型学习率，优化器
    '''
    # Step 2: prepare model, criterion, optimizer, and learning rate scheduler.
    net = getattr(models, args.arch)(num_classes=10).to(device) # 神经网络模型
    criterion = torch.nn.CrossEntropyLoss().to(device) # 损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4) # 优化器
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1) # 学习率

    '''
    训练后门模型
    '''
    # Step 3: train backdoored models
    logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_init.th')) # 初始化一个模型文件
    if trigger_info is not None:
        torch.save(trigger_info, os.path.join(args.output_dir, 'trigger_info.th')) # 初始化触发器文件
    for epoch in range(1, args.epoch): # 每个轮次循环
        start = time.time()
        lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train(model=net, criterion=criterion, optimizer=optimizer,
                                      data_loader=poison_train_loader) # 调用一次训练函数，进行一个轮次的迭代
        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader) # 用干净测试集测试一次
        po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader) # 用中毒测试集测试一次
        scheduler.step()
        end = time.time()
        logger.info(
            '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc)

        if (epoch + 1) % args.save_every == 0:
            torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_{}.th'.format(epoch))) # 将训练后的检查点进行保存

    # save the last checkpoint
    torch.save(net.state_dict(), os.path.join(args.output_dir, 'model_last.th')) # 将最后一次训练结果保存

'''
训练模型
'''
def train(model, criterion, optimizer, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc

'''
测试模型
'''
def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


if __name__ == '__main__':
    main()
