from models.MGOD.modules import UNet3D, modality_muaware_block, get_Mutiaxis_decoder_shuffle_fea
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.losses import dice_loss
import numpy as np


def get_rand_miss():
    random_miss = torch.tensor(np.random.binomial(n=1, p=0.5, size=4))
    random_miss = random_miss.reshape([1, 4, 1, 1, 1])
    if torch.sum(random_miss) == 0:
        random_miss = get_rand_miss()
    if torch.sum(random_miss) == 4:
        random_miss = get_rand_miss()
    return random_miss


class Multi_Granularity_Online_Distillation(nn.Module):
    def __init__(self):
        super(Multi_Granularity_Online_Distillation, self).__init__()
        self.unet = UNet3D()
        self.ema_model = UNet3D()
        self.MMD = modality_muaware_block()
        self.Mutiaxis_fuse2 = get_Mutiaxis_decoder_shuffle_fea(in_dim=16, out_dim=256, dowm_scale=1)
        self.Mutiaxis_fuse3 = get_Mutiaxis_decoder_shuffle_fea(in_dim=32, out_dim=128, dowm_scale=2)
        self.Mutiaxis_fuse4 = get_Mutiaxis_decoder_shuffle_fea(in_dim=64, out_dim=64, dowm_scale=4)

    def forward(self, x, batch_y):
        # x: default shape [1, 4, 160, 192, 128]
        # batch_y: default shape [1, 3, 160, 192, 128]
        random_miss = get_rand_miss().cuda()

        x_full = x  # default ["t1", "t2", "flair", "t1ce"]
        x_miss = x * random_miss

        with torch.no_grad():
            full_out_t = self.ema_model(x_full)[0]

        full_out, full_style, f_u2, f_u3, f_u4 = self.unet(x_full)

        miss_out, miss_style, m_u2, m_u3, m_u4 = self.unet(x_miss)

        MMD_loss = self.MMD(full_style, miss_style, x_full, random_miss)
        del miss_style, full_style, x_full, x_miss

        full_fea2, miss_fea2 = self.Mutiaxis_fuse2(f_u2, m_u2)
        full_fea3, miss_fea3 = self.Mutiaxis_fuse3(f_u3, m_u3)
        full_fea4, miss_fea4 = self.Mutiaxis_fuse4(f_u4, m_u4)

        del f_u2, f_u3, f_u4, m_u2, m_u3, m_u4

        feature_list = [full_out, miss_out, full_fea2, miss_fea2, full_fea3, miss_fea3, full_fea4, miss_fea4,
                        full_out_t, MMD_loss]

        loss_dict = compute_loss(feature_list, batch_y)

        return loss_dict, full_out, miss_out

    def val_forward(self, x_miss):
        miss_out = self.unet(x_miss)[0]
        return miss_out

    def get_model(self):
        return self.unet


def compute_loss(feature_list, label):
    [full_out, miss_out, full_fea2, miss_fea2, full_fea3, miss_fea3, full_fea4, miss_fea4, full_out_t,
     MMD_loss] = feature_list

    loss_dict = {}
    loss_dict['ed_full_loss'] = dice_loss(full_out[:, 0], label[:, 0])
    loss_dict['net_full_loss'] = dice_loss(full_out[:, 1], label[:, 1])
    loss_dict['et_full_loss'] = dice_loss(full_out[:, 2], label[:, 2])

    loss_dict['ed_miss_loss'] = dice_loss(miss_out[:, 0], label[:, 0])
    loss_dict['net_miss_loss'] = dice_loss(miss_out[:, 1], label[:, 1])
    loss_dict['et_miss_loss'] = dice_loss(miss_out[:, 2], label[:, 2])

    loss_dict['sup_full_loss'] = loss_dict['ed_full_loss'] + loss_dict['net_full_loss'] + loss_dict['et_full_loss']
    loss_dict['sup_miss_loss'] = loss_dict['ed_miss_loss'] + loss_dict['net_miss_loss'] + loss_dict['et_miss_loss']

    loss_dict['feature_level_loss'] = (F.mse_loss(full_fea2[:, :], miss_fea2[:, :], reduction='mean') +
                                       F.mse_loss(full_fea3[:, :], miss_fea3[:, :], reduction='mean') +
                                       F.mse_loss(full_fea4[:, :], miss_fea4[:, :], reduction='mean'))

    loss_dict['image_level_loss'] = (F.mse_loss(full_out[:, :], full_out_t[:, :], reduction='mean') +
                                     F.mse_loss(miss_out[:, :], full_out_t[:, :], reduction='mean'))

    loss_dict['modality_level_loss'] = MMD_loss

    weight_missing = 1
    weight_full = 1

    weight1 = 10
    weight2 = 150
    weight3 = 50

    loss_dict['loss_total'] = (weight_missing * loss_dict['loss_miss_dc'] + loss_dict['loss_dc'] * weight_full +
                            weight1 * loss_dict['out_consistency_loss'] + weight2 * MMD_loss
                            + loss_dict['mutiaxis_align_loss'] * weight3)

    return loss_dict