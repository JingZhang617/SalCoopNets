import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from datetime import datetime

from model.ResNet_models import Generator, Descriptor
from data import get_loader
from utils import clip_gradient, adjust_lr, AvgMeter
from scipy import misc
import cv2
import torchvision.transforms as transforms



parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=30, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=5e-5, help='learning rate')
parser.add_argument('--lr_des', type=float, default=1e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--des_sim_weight', type=float, default=0.3, help='descrimitor similarity weight')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=20, help='every n epochs decay learning rate')
parser.add_argument('-beta1_gen', type=float, default=0.5,help='beta of Adam for generator')
parser.add_argument('-beta1_des', type=float, default=0.5,help='beta of Adam for descriptor')

parser.add_argument('--latent_dim', type=int, default=8, help='latent dim')
parser.add_argument('--channel_reduced_gen', type=int, default=32, help='reduced channel dimension for generator')

parser.add_argument('--langevin_step_num_gen', type=float, default=3, help='langevin step num of generator')
parser.add_argument('--sigma_gen', type=float, default=0.3, help='sigma for generator')
parser.add_argument('--langevin_s', type=float, default=0.1, help='s in langevin sampling')

parser.add_argument('--langevin_step_num_des', type=int, default=3, help='number of langevin steps for ebm')
parser.add_argument('-langevin_step_size_des', type=float, default=0.001,help='step size of EBM langevin')
parser.add_argument('-sigma_des', type=float, default=0.1, help='sigma of EBM langevin')
parser.add_argument('--energy_form', default='identity', help='tanh | sigmoid | identity | softplus')
parser.add_argument('--channel_reduced_des', type=int, default=64, help='reduced channel dimension for descriptor')

opt = parser.parse_args()

print('Generator Learning Rate: {}'.format(opt.lr_gen))
print('Descriptor Learning Rate: {}'.format(opt.lr_des))
# build models
generator = Generator(channel=opt.channel_reduced_gen, latent_dim=opt.latent_dim)
descriptor = Descriptor(channel=opt.channel_reduced_des)

generator.cuda()
descriptor.cuda()
generator_params = generator.parameters()
descriptor_params = descriptor.parameters()
generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen)
descriptor_optimizer = torch.optim.Adam(descriptor_params, opt.lr_des)


image_root = './train/DUTS/img/'
gt_root = './train/DUTS/gt/'

train_loader, training_set_size = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)
train_z = torch.FloatTensor(training_set_size, opt.latent_dim).normal_(0, 1).cuda()

CE = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
size_rates = [0.75,1,1.25]  # multi-scale training

save_path = './temp/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

def visualize_uncertainty1(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        name = '{:02d}_gen_final.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_uncertainty2(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        name = '{:02d}_des.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_uncertainty3(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        name = '{:02d}_gen_ref.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_uncertainty4(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        name = '{:02d}_gen_init.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_gt(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        name = '{:02d}_gt.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

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

def min_max_norm(in_):
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_-min_+1e-8)

# linear annealing to avoid posterior collapse
def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)

    return annealed

print("Let's go!")
for epoch in range(1, opt.epoch+1):
    generator.train()
    descriptor.train()
    loss_record_gen = AvgMeter()
    loss_record_des = AvgMeter()
    print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))
    print('Descriptor Learning Rate: {}'.format(descriptor_optimizer.param_groups[0]['lr']))

    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            generator_optimizer.zero_grad()
            descriptor_optimizer.zero_grad()
            images, gts, index_batch = pack
            images = Variable(images)
            gts = Variable(gts)
            images = images.cuda()
            gts = gts.cuda()

            # multi-scale training samples
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            ## G0: obtain generator prediction
            z_noise = torch.randn(images.shape[0], opt.latent_dim).cuda()
            z_noise = Variable(z_noise.cuda(), requires_grad=True)
            _, ref_generator = generator(images, z_noise)

            ## D1: revise generator prediction with descriptor
            seg = ref_generator.detach()
            gen_preds = [seg.clone() for _ in range(opt.langevin_step_num_des + 1)]
            for kk in range(opt.langevin_step_num_des):
                pred_seg = Variable(gen_preds[kk], requires_grad=True)
                pred_seg = pred_seg.cuda()
                noise = torch.FloatTensor(pred_seg.size()).normal_(0, 1).cuda()
                joint_energy = compute_energy(descriptor(images, torch.sigmoid(pred_seg)))
                joint_energy.backward(torch.ones(joint_energy.size()).cuda())
                pred_seg_grad = pred_seg.grad
                pred_seg = pred_seg - 0.5 * opt.langevin_step_size_des * opt.langevin_step_size_des * pred_seg_grad
                if epoch < 20:
                    pred_seg += opt.langevin_step_size_des * noise
                gen_preds[kk + 1] = pred_seg

            revised_pred = gen_preds[-1]
            # revised_pred = min_max_norm(revised_pred)
            revised_pred1 = torch.sigmoid(revised_pred).detach()

            ## G1: update latent variable z of generator with MCMC
            anneal_reg = linear_annealing(0, 1, epoch, opt.epoch)

            z_noise_preds = [z_noise.clone() for _ in range(opt.langevin_step_num_gen + 1)]
            for kk in range(opt.langevin_step_num_gen):
                z_noise = Variable(z_noise_preds[kk], requires_grad=True)
                z_noise = z_noise.cuda()
                noise = torch.randn(z_noise.size())
                noise = noise.cuda()
                gen_ini, gen_res = generator(images, z_noise)
                gen_loss1_1 = opt.des_sim_weight / (2.0 * opt.sigma_gen * opt.sigma_gen) * mse_loss(torch.sigmoid(gen_res),revised_pred1)
                gen_loss1_2 = opt.des_sim_weight / (2.0 * opt.sigma_gen * opt.sigma_gen) * mse_loss(torch.sigmoid(gen_ini),
                                                                                                    revised_pred1)
                gen_loss2_1 = (1-opt.des_sim_weight)/ (2.0 * opt.sigma_gen * opt.sigma_gen) * structure_loss(gen_res, gts)
                gen_loss2_2 = (1 - opt.des_sim_weight) / (2.0 * opt.sigma_gen * opt.sigma_gen) * structure_loss(gen_ini,
                                                                                                                gts)
                gen_loss1 = anneal_reg * gen_loss1_1 + (1-anneal_reg)*gen_loss2_1
                gen_loss2 = anneal_reg * gen_loss1_2 + (1-anneal_reg)*gen_loss2_2
                gen_loss = gen_loss1 + gen_loss2
                gen_loss.backward(torch.ones(gen_loss.size()).cuda())
                grad = z_noise.grad
                z_noise = z_noise - 0.5 * opt.langevin_s * opt.langevin_s * (z_noise + grad)
                z_noise += opt.langevin_s * noise
                z_noise_preds[kk + 1] = z_noise
            z_noise = z_noise_preds[-1]

            ## D2: define descriptor loss, and update descriptor
            revised_pred = torch.sigmoid(revised_pred)
            obs_feature = descriptor(images, gts)
            obs_eng = compute_energy(obs_feature)
            E_obs = torch.mean(obs_eng)
            revised_feature = descriptor(images, revised_pred)

            revised_eng = compute_energy(revised_feature)
            E_pred = torch.mean(revised_eng)
            des_loss = E_obs - E_pred
            des_loss.backward()
            descriptor_optimizer.step()

            ## G2: update generator with updated latent variable z
            ini_generator, generator_pred_final = generator(images,z_noise)
            gen_loss = anneal_reg*(mse_loss(torch.sigmoid(generator_pred_final), revised_pred.detach()) + \
                       mse_loss(torch.sigmoid(ini_generator), revised_pred.detach())) + \
                       (1-anneal_reg)*(structure_loss(generator_pred_final, gts) + \
                       structure_loss(ini_generator, gts))
            gen_loss.backward()
            generator_optimizer.step()

            ## visalize predictions
            visualize_uncertainty1(torch.sigmoid(generator_pred_final))
            visualize_uncertainty3(torch.sigmoid(ref_generator))
            visualize_uncertainty4(torch.sigmoid(ini_generator))
            visualize_uncertainty2(revised_pred)
            visualize_gt(gts)

            if rate == 1:
                loss_record_gen.update(gen_loss.data, opt.batchsize)
                loss_record_des.update(des_loss.data, opt.batchsize)

        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Gen Loss: {:.4f}, Des Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record_gen.show(), loss_record_des.show()))

    adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)
    adjust_lr(descriptor_optimizer, opt.lr_des, epoch, opt.decay_rate, opt.decay_epoch)

    save_path = 'models/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch % 10 == 0:
        torch.save(generator.state_dict(), save_path + 'Model' + '_%d' % epoch + '_gen.pth')
        torch.save(descriptor.state_dict(), save_path + 'Model' + '_%d' % epoch + '_des.pth')
