import torch
from torch import nn
from sklearn.metrics import mean_squared_error
import os
import sys
import networks
from networks import UNet

class Pix2Pix(nn.Module):
    def __init__(self, config):
        super(Pix2Pix, self).__init__()

        self.config = config
        self.device = torch.device('cuda:' + str(self.config.gpu_ids[0]) if torch.cuda.is_available() else 'cpu')

        # self.loss_names = ['G_GAN', 'G_l1', 'D_real', 'D_fake']
        # self.visual_names = ['input', 'fake_output', 'real_output']

        self.netG = UNet().to(self.device)
        self.netD = networks.define_D(self.config.input_nc + self.config.output_nc, self.config.ndf, self.config.netD,
                                      self.config.n_layers_D, self.config.norm, self.config.init_type, self.config.init_gain, self.config.gpu_ids)

        self.criterion_GAN = networks.GANLoss(config.gan_mode).to(self.device)
        self.criterion_L1 = torch.nn.L1Loss()

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.999))
        self.G_scheduler = networks.get_scheduler(self.optimizer_G, config)
        self.D_scheduler = networks.get_scheduler(self.optimizer_D, config)

        self.G_loss = 0
        self.G_loss_GAN = 0
        self.G_loss_L1 = 0
        self.D_loss = 0

    ## functions
    def train(self, idx, data):
        input = data['input'].type(torch.FloatTensor).to(self.device)
        real_label = data['label'].type(torch.FloatTensor).to(self.device)

        ### forward ###
        # Generator
        fake_label = self.netG(input)

        # Discriminator - fake
        D_fake_in = torch.cat((fake_label, input), dim=1)
        D_fake_out = self.netD(D_fake_in.detach())   # torch.Size([batch, 1, 6, 3])

        # Discriminator - real
        D_real_in = torch.cat((real_label, input), dim=1)
        D_real_out = self.netD(D_real_in.detach())

        ### backward ###
        # Loss - G
        self.optimizer_G.zero_grad()
        self.G_loss_GAN = self.criterion_GAN(D_fake_out, True)
        self.G_loss_L1 = self.criterion_L1(fake_label, real_label)

        self.G_loss = self.G_loss_GAN + self.config.lambda_L1 * self.G_loss_L1
        self.G_loss.backward(retain_graph=True)
        self.optimizer_G.step()

        # Loss - D
        self.optimizer_D.zero_grad()
        D_loss_fake = self.criterion_GAN(D_fake_out, False)
        D_loss_real = self.criterion_GAN(D_real_out, True)
        self.D_loss = (D_loss_fake + D_loss_real) / 2
        self.D_loss.backward()
        self.optimizer_D.step()

        train_dict = {}
        train_dict['G_loss'] = self.G_loss
        train_dict['G_GAN_loss'] = self.G_loss_GAN
        train_dict['G_L1_loss'] =self. G_loss_L1
        train_dict['D_loss'] = self.D_loss
        train_dict['fake_label'] = fake_label
        train_dict['real_label'] = real_label

        return train_dict

    def val(self, data):
        with torch.no_grad():
            input = data['input'].type(torch.FloatTensor).to(self.device)
            real_label = data['label'].type(torch.FloatTensor).to(self.device)

            # Generator
            fake_label = self.netG(input)  # Tensor (batch, 1, 66, 45)

            # Discriminator - fake
            D_fake_in = torch.cat((fake_label, input), dim=1)
            D_fake_out = self.netD(D_fake_in.detach())  # torch.Size([batch, 1, 6, 3])

            # Discriminator - real
            D_real_in = torch.cat((real_label, input), dim=1)
            D_real_out = self.netD(D_real_in.detach())

            ### backward ###
            # Loss - G
            G_loss_GAN = self.criterion_GAN(D_fake_out, True)
            G_loss_L1 = self.criterion_L1(fake_label, real_label)

            G_loss = G_loss_GAN + self.config.lambda_L1 * G_loss_L1

            # Loss - D
            D_loss_fake = self.criterion_GAN(D_fake_out, False)
            D_loss_real = self.criterion_GAN(D_real_out, True)
            D_loss = (D_loss_fake + D_loss_real) / 2

            val_dict = {}
            val_dict['G_loss'] = G_loss
            val_dict['D_loss'] = D_loss
            val_dict['fake_label'] = fake_label
            val_dict['real_label'] = real_label

        return val_dict

    def test(self, data):
        with torch.no_grad():
            input = data['input'].type(torch.FloatTensor).to(self.device)
            sub = data['sub']
            # Generator
            fake_label = self.netG(input)  # Tensor (batch_size, 1, 66, 45)
            test_dict = {}
            test_dict['fake_label'] = fake_label
            test_dict['sub'] = sub
        return test_dict

    def forward(self, x):
        fake_depth = self.netG(x)
        D_fake_in = torch.cat((fake_depth.detach().cpu(), x), dim=1)
        result = self.netD(D_fake_in)
        return result