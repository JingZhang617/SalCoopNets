import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
from scipy import misc

from model.ResNet_models import Generator, Descriptor
from data import test_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--latent_dim', type=int, default=6, help='latent dim')
parser.add_argument('--channel_reduced_gen', type=int, default=32, help='reduced channel dimension for generator')
parser.add_argument('--channel_reduced_des', type=int, default=64, help='reduced channel dimension for descriptor')
parser.add_argument('--energy_form', default='identity', help='tanh | sigmoid | identity | softplus')
parser.add_argument('-sigma_des', type=float, default=0.1,help='sigma of EBM langevin')
parser.add_argument('--langevin_step_num_des', type=int, default=3, help='number of langevin steps for ebm')
parser.add_argument('-langevin_step_size_des', type=float, default=0.001,help='step size of EBM langevin')
parser.add_argument('--z_sample_iterations', type=int, default=10, help='number of iterations for sampling z from latent space')
opt = parser.parse_args()

# dataset_path = '/home/jing-zhang/jing_file/RGB_sal_dataset/train/DUTS/img/'
dataset_path = '/home/jingzhang/jing_files/RGB_Dataset/test/img/'
gt_path = '/home/jingzhang/jing_files/RGB_Dataset/test/gt/'


generator = Generator(channel=opt.channel_reduced_gen, latent_dim=opt.latent_dim)
descriptor = Descriptor(channel=opt.channel_reduced_des)
generator.load_state_dict(torch.load('./models/Resnet/Model_30_gen.pth'))
descriptor.load_state_dict(torch.load('./models/Resnet/Model_30_des.pth'))


generator.cuda()
generator.eval()
descriptor.cuda()
descriptor.eval()
test_datasets = ['ECSSD','DUT','DUTS_Test','HKU-IS', 'PASCAL', 'SOD']

def compute_energy(disc_score):
    if opt.energy_form == 'tanh':
        energy = torch.tanh(-disc_score.squeeze())
    elif opt.energy_form == 'sigmoid':
        energy = F.sigmoid(-disc_score.squeeze())
    elif opt.energy_form == 'identity':
        energy = -disc_score.squeeze()
    elif opt.energy_form == 'softplus':
        energy = F.softplus(-disc_score.squeeze())
    return energy

for dataset in test_datasets:
    save_path_mean = './results_mean/' + dataset + '/'
    save_path_var = './results_var/' + dataset + '/'
    if not os.path.exists(save_path_mean):
        os.makedirs(save_path_mean)
    if not os.path.exists(save_path_var):
        os.makedirs(save_path_var)

    image_root = dataset_path + dataset + '/'
    test_loader = test_dataset(image_root, opt.testsize)
    for i in range(test_loader.size):
        print(i)
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        sal_pred = list()
        for i in range(10):
            z_noise = torch.zeros(image.shape[0], opt.latent_dim).cuda()
            _, generator_pred = generator.forward(image, z_noise)
            seg1 = generator_pred
            temp = seg1
            sal_pred.append(torch.sigmoid(temp))
        sal_preds = sal_pred[0].clone()
        for iter in range(1, 10):
            sal_preds = torch.cat((sal_preds, sal_pred[iter]), 1)
        mean_pred = torch.mean(sal_preds, dim=1, keepdim=True)
        var = -mean_pred * torch.log(mean_pred + 1e-8)
        res = mean_pred
        res = F.upsample(res, size=[WW, HH], mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path_mean + name, res)

        res = var
        res = F.upsample(res, size=[WW, HH], mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path_var + name, res)

        # z_noise = torch.zeros(image.shape[0], opt.latent_dim).cuda()
        # _, generator_pred = generator.forward(image,z_noise)
        # seg1 = generator_pred
        # gen_preds1 = [seg1.clone() for _ in range(opt.langevin_step_num_des + 1)]
        # for kk in range(opt.langevin_step_num_des):
        #     pred_seg = Variable(gen_preds1[kk], requires_grad=True)
        #     pred_seg = pred_seg.cuda()
        #     joint_energy = compute_energy(descriptor.forward(image, torch.sigmoid(pred_seg)))
        #     joint_energy.backward(torch.ones(joint_energy.size()).cuda())
        #     pred_seg_grad = pred_seg.grad
        #     pred_seg = pred_seg - 0.5 * opt.langevin_step_size_des * opt.langevin_step_size_des * pred_seg_grad
        #     gen_preds1[kk + 1] = pred_seg
        # res = generator_pred
        # res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
        # res = res.sigmoid().data.cpu().numpy().squeeze()
        # #res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # cv2.imwrite(save_path+name, res)
