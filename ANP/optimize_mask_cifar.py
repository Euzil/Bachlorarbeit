import os
import time
import argparse
import numpy as np
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import models
import data.poison_cifar as poison

parser = argparse.ArgumentParser(description='Train poisoned networks')

# Basic model parameters.
parser.add_argument('--arch', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2', 'vgg19_bn'])
parser.add_argument('--checkpoint', type=str, default='./save/last_model.th', help='The checkpoint to be pruned')
parser.add_argument('--widen-factor', type=int, default=1, help='widen_factor for WideResNet')
parser.add_argument('--batch-size', type=int, default=128, help='the batch size for dataloader')
parser.add_argument('--lr', type=float, default=0.2, help='the learning rate for mask optimization')
parser.add_argument('--nb-iter', type=int, default=2000, help='the number of iterations for training')
parser.add_argument('--print-every', type=int, default=500, help='print results every few iterations')
parser.add_argument('--data-dir', type=str, default='../data', help='dir to the dataset')
parser.add_argument('--val-frac', type=float, default=0.01, help='The fraction of the validate set')
parser.add_argument('--output-dir', type=str, default='logs/models/')

parser.add_argument('--trigger-info', type=str, default='', help='The information of backdoor trigger')
parser.add_argument('--poison-type', type=str, default='benign', choices=['badnets', 'blend', 'clean-label', 'benign'],
                    help='type of backdoor attacks for evaluation')
parser.add_argument('--poison-target', type=int, default=0, help='target class of backdoor attack')
parser.add_argument('--trigger-alpha', type=float, default=1.0, help='the transparency of the trigger pattern.')

parser.add_argument('--anp-eps', type=float, default=0.4)
parser.add_argument('--anp-steps', type=int, default=1)
parser.add_argument('--anp-alpha', type=float, default=0.2)

args = parser.parse_args()
args_dict = vars(args)
print(args_dict)
os.makedirs(args.output_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
第二步，优化一个掩模
'''
def main():
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
    创建一个数据集，干净训练集，中毒测试集，干净测试集
    '''
    # Step 1: create dataset - clean val set, poisoned test set, and clean test set. 
    if args.trigger_info:
        trigger_info = torch.load(args.trigger_info, map_location=device)
    else:
        if args.poison_type == 'benign':
            trigger_info = None
        else:
            triggers = {'badnets': 'checkerboard_1corner',
                        'clean-label': 'checkerboard_4corner',
                        'blend': 'gaussian_noise'}
            trigger_type = triggers[args.poison_type]
            pattern, mask = poison.generate_trigger(trigger_type=trigger_type) # 生成触发器
            trigger_info = {'trigger_pattern': pattern[np.newaxis, :, :, :], 'trigger_mask': mask[np.newaxis, :, :, :],
                            'trigger_alpha': args.trigger_alpha, 'poison_target': np.array([args.poison_target])}

    orig_train = CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train) # 原始数据集
    _, clean_val = poison.split_dataset(dataset=orig_train, val_frac=args.val_frac, 
                                        perm=np.loadtxt('./data/cifar_shuffle.txt', dtype=int)) # 分割数据集
    clean_test = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test) # 干净测试数据集
    poison_test = poison.add_predefined_trigger_cifar(data_set=clean_test, trigger_info=trigger_info) # 中毒测试数据集

    random_sampler = RandomSampler(data_source=clean_val, replacement=True,
                                   num_samples=args.print_every * args.batch_size) # 随机采样
    clean_val_loader = DataLoader(clean_val, batch_size=args.batch_size,
                                  shuffle=False, sampler=random_sampler, num_workers=0) # 加载干净数据
    poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=0) # 加载中毒测试数据
    clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=0) # 加载干净测试数据
    '''
    加载模型和检查点
    '''
    # Step 2: load model checkpoints and trigger info
    state_dict = torch.load(args.checkpoint, map_location=device) # 加载当前状态
    net = getattr(models, args.arch)(num_classes=10, norm_layer=models.NoisyBatchNorm2d) # 加载网络
    load_state_dict(net, orig_state_dict=state_dict) # 加载
    net = net.to(device) # 神经网络
    criterion = torch.nn.CrossEntropyLoss().to(device) # 损失函数

    parameters = list(net.named_parameters()) # 给出网络层的名字和参数的迭代器
    mask_params = [v for n, v in parameters if "neuron_mask" in n] # 设置掩膜的参数
    mask_optimizer = torch.optim.SGD(mask_params, lr=args.lr, momentum=0.9) # 优化掩膜，会把数据拆分后再分批不断放入 NN 中计算
    noise_params = [v for n, v in parameters if "neuron_noise" in n] # 设置神经元扰动的参数
    noise_optimizer = torch.optim.SGD(noise_params, lr=args.anp_eps / args.anp_steps) # 优化神经元扰动
    '''
    训练后门模型
    '''
    # Step 3: train backdoored models
    print('Iter \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    nb_repeat = int(np.ceil(args.nb_iter / args.print_every))
    for i in range(nb_repeat):
        start = time.time()
        lr = mask_optimizer.param_groups[0]['lr']
        train_loss, train_acc = mask_train(model=net, criterion=criterion, data_loader=clean_val_loader,
                                           mask_opt=mask_optimizer, noise_opt=noise_optimizer) # 训练掩膜
        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader) # 干净测试
        po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader) # 中毒测试
        end = time.time()
        print('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            (i + 1) * args.print_every, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc))
    save_mask_scores(net.state_dict(), os.path.join(args.output_dir, 'mask_values.txt')) # 保存掩膜


def load_state_dict(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']
    if "state_dict" in orig_state_dict.keys():
        orig_state_dict = orig_state_dict["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        elif 'running_mean_noisy' in k or 'running_var_noisy' in k or 'num_batches_tracked_noisy' in k:
            new_state_dict[k] = orig_state_dict[k[:-6]].clone().detach()
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)


def clip_mask(model, lower=0.0, upper=1.0):
    params = [param for name, param in model.named_parameters() if 'neuron_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def sign_grad(model):
    noise = [param for name, param in model.named_parameters() if 'neuron_noise' in name]
    for p in noise:
        p.grad.data = torch.sign(p.grad.data)


def perturb(model, is_perturbed=True):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
            module.perturb(is_perturbed=is_perturbed)


def include_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
            module.include_noise()


def exclude_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
            module.exclude_noise()


def reset(model, rand_init):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
            module.reset(rand_init=rand_init, eps=args.anp_eps)


def mask_train(model, criterion, mask_opt, noise_opt, data_loader):
    model.train()
    total_correct = 0 # 初始化正确的神经元
    total_loss = 0.0 # 初始化错误的神经元
    nb_samples = 0 # 初始化采样数
    for i, (images, labels) in enumerate(data_loader):  # 枚举干净训练集
        images, labels = images.to(device), labels.to(device)
        nb_samples += images.size(0)

        # step 1: calculate the adversarial perturbation for neurons
        if args.anp_eps > 0.0:
            reset(model, rand_init=True)
            for _ in range(args.anp_steps): # 循环优化扰动
                noise_opt.zero_grad() # 梯度下降优化神经元扰动

                include_noise(model) # 加入扰动
                output_noise = model(images) # 
                loss_noise = - criterion(output_noise, labels)

                loss_noise.backward()
                sign_grad(model)
                noise_opt.step()

        # step 2: calculate loss and update the mask values
        mask_opt.zero_grad()
        if args.anp_eps > 0.0:
            include_noise(model)
            output_noise = model(images)
            loss_rob = criterion(output_noise, labels)
        else:
            loss_rob = 0.0

        exclude_noise(model)
        output_clean = model(images)
        loss_nat = criterion(output_clean, labels)
        loss = args.anp_alpha * loss_nat + (1 - args.anp_alpha) * loss_rob

        pred = output_clean.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()
        mask_opt.step()
        clip_mask(model)

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc


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


def save_mask_scores(state_dict, file_name):
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if 'neuron_mask' in name:
            for idx in range(param.size(0)):
                neuron_name = '.'.join(name.split('.')[:-1])
                mask_values.append('{} \t {} \t {} \t {:.4f} \n'.format(count, neuron_name, idx, param[idx].item()))
                count += 1
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask Score \n')
        f.writelines(mask_values)


if __name__ == '__main__':
    main()
