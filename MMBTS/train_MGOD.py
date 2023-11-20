import os
import torch
import torch.optim as optim
import numpy as np

from models.discriminator import get_style_discriminator
from models.MGOD.mgsd import Multi_Granularity_Online_Distillation


def training(model, d_style, loaders, optimizer, scheduler, epoch_init=0):

    n_epochs = 300
    iter_num = 0
    phase = "train"
    lr = float(1e-4)

    for epoch in range(epoch_init, n_epochs):
        epoch = epoch + 1
        scheduler.step()

        loader = loaders[phase]
        for batch_id, (batch_x, batch_y) in enumerate(loader):
            iter_num = iter_num + 1
            batch_x, batch_y = batch_x.cuda(non_blocking=True), batch_y.cuda(non_blocking=True)
            with torch.set_grad_enabled(phase == 'train'):

                loss_dict, full_out, miss_out = model(batch_x, batch_y)

                d_style.train()
                optimizer_d_style = optim.Adam(d_style.parameters(), lr=lr, betas=(0.9, 0.99))
                source_label = 0
                target_label = 1

                optimizer.zero_grad()
                optimizer_d_style.zero_grad()

                for param in d_style.parameters():
                    param.requires_grad = False

                (loss_dict['loss_total']).backward(retain_graph=True)
                # EMA
                model.ema_model = update_ema_variables(model.ema_model, model.unet)

                # Generative adversarial training
                df_src_main = full_out
                df_trg_main = miss_out
                d_df_out_main = d_style(df_trg_main)
                loss_adv_df_trg_main = bce_loss(d_df_out_main, source_label)
                loss = loss_adv_df_trg_main
                loss.backward()

                for param in d_style.parameters():
                    param.requires_grad = True

                df_src_main = df_src_main.detach()
                d_df_out_main = d_style(df_src_main)
                loss_d_feature_main = bce_loss(d_df_out_main, source_label)
                loss_d_feature_main.backward()

                df_trg_main = df_trg_main.detach()
                d_df_out_main = d_style(df_trg_main)
                loss_d_feature_main = bce_loss(d_df_out_main, target_label)
                loss_d_feature_main.backward()

            print("In Epoch ", epoch, " batch ", batch_id, " ", loss_dict['loss_total'])

            optimizer.step()
            optimizer_d_style.step()


def train_proccess():

    model = Multi_Granularity_Online_Distillation().cuda()
    d_style = get_style_discriminator(num_classes=3).cuda()

    loaders = make_data_loaders(train_txt, val_txt)
    optimizer, scheduler = make_optimizer(model)

    training(model, d_style, loaders, optimizer, scheduler)


if __name__ == '__main__':
    train_proccess()
