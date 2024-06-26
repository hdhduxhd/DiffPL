import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools
import sys
sys.path[0]='/kaggle/working/DiffPL'
from CycleGAN.util.image_pool import ImagePool
from CycleGAN.models.base_model import BaseModel
from CycleGAN.models import networks
from ddpm.new_diffusion import *
from ddpm.unet import UNet

# -ln(circularity)
def circularity(y_pred):
    """
    y_pred: BxHxW
    """
    x = y_pred[:, 1:, :] - y_pred[:, :-1, :]  # horizontal and vertical directions
    y = y_pred[:, :, 1:] - y_pred[:, :, :-1]

    delta_x = x[:, :, 1:] ** 2
    delta_y = y[:, 1:, :] ** 2
    delta_u = torch.abs(delta_x + delta_y)

    epsilon = 0.00000001  # a small value to avoid division by zero in practice
    length = torch.sqrt(delta_u + epsilon).sum(dim=[1, 2])
    area = y_pred.sum(dim=[1, 2])

    compactness_loss = torch.sum(-1 * torch.log((area * 4 * 3.1415926 + epsilon) / (length ** 2)))
    return compactness_loss

# 1/circularity
# def circularity(y_pred):
#     """
#     y_pred: BxHxW
#     """
#     x = y_pred[:, 1:, :] - y_pred[:, :-1, :]  # horizontal and vertical directions
#     y = y_pred[:, :, 1:] - y_pred[:, :, :-1]

#     delta_x = x[:, :, 1:] ** 2
#     delta_y = y[:, 1:, :] ** 2
#     delta_u = torch.abs(delta_x + delta_y)

#     epsilon = 0.00000001  # a small value to avoid division by zero in practice
#     w = 0.01
#     length = w * torch.sqrt(delta_u + epsilon).sum(dim=[1, 2])
#     area = y_pred.sum(dim=[1, 2])

#     compactness_loss = torch.sum(length ** 2 / (area * 4 * 3.1415926))
#     return compactness_loss

def circularity_2label(y_pred):
    return (circularity(y_pred[:,0,...]) + circularity(y_pred[:,1,...])) / 2

class DiffCycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.add_argument('--max_timestep', type=int, default=500, help='output dim of linear')
        parser.add_argument('--shift', type=int, default=0, help='start index of timestep')
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_N', type=float, default=20.0, help='weight for diff loss (denoise)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'diff']
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'circularity']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'real_A_noise', 'fake_B_noise']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'real_B_noise', 'fake_A_noise']
        # if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
        #     visual_names_A.append('idt_B')
        #     visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            # self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'Denoise_A', 'Denoise_B']
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'G_N']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B', 'G_N']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_N = networks.define_N(opt.input_nc, opt.ngf, opt.n_layers_N, opt.max_timestep, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netG_N_cup = networks.define_N(opt.input_nc, opt.ngf, opt.n_layers_D, opt.max_timestep, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netG_N_disc = networks.define_N(opt.input_nc, opt.ngf, opt.n_layers_D, opt.max_timestep, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.diffusion = GaussianDiffusion()
        # self.netDenoise_A = UNet(in_channel=opt.input_nc, out_channel=opt.input_nc).to(self.device)
        # self.netDenoise_B = UNet(in_channel=opt.output_nc, out_channel=opt.output_nc).to(self.device)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(), self.netDenoise_A.parameters(), self.netDenoise_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(), self.netG_N.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # logits = self.netG_N(self.real_B)
        logits1, logits2, logits3 = self.netG_N(self.real_B)
        # y = get_rep_outputs(logits, 0.5, True)
        y1 = get_rep_outputs(logits1, 0.5, True)
        y2 = get_rep_outputs(logits2, 0.5, True)
        y3 = get_rep_outputs(logits3, 0.5, True)
        # column_vector = torch.arange(self.opt.shift+1, self.opt.shift+self.opt.max_timestep+1).view(self.opt.max_timestep, 1).cuda()
        column_vector1 = torch.arange(0, self.opt.max_timestep//100).view(self.opt.max_timestep//100, 1).cuda()
        column_vector2 = torch.arange(0, 10).view(10, 1).cuda()
        column_vector3 = torch.arange(1, 10).view(9, 1).cuda()
        # self.t = (y1.permute(0,2,3,1) @ column_vector1.float()) * 100 + (y2.permute(0,2,3,1) @ column_vector2.float()) * 10 + (y3.permute(0,2,3,1) @ column_vector3.float())
        self.t = (y1 @ column_vector1.float()) * 100 + (y2 @ column_vector2.float()) * 10 + (y3 @ column_vector3.float())

        # logits1, logits2, logits3 = self.netG_N_disc(self.real_B)
        # y1 = get_rep_outputs(logits1, 0.5, True)
        # y2 = get_rep_outputs(logits2, 0.5, True)
        # y3 = get_rep_outputs(logits3, 0.5, True)
        # column_vector1 = torch.arange(0, self.opt.max_timestep//100).view(self.opt.max_timestep//100, 1).cuda()
        # column_vector2 = torch.arange(0, 10).view(10, 1).cuda()
        # column_vector3 = torch.arange(1, 10).view(9, 1).cuda()
        # self.t_disc = (y1 @ column_vector1.float()) * 100 + (y2 @ column_vector2.float()) * 10 + (y3 @ column_vector3.float())
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']
    
    def forward(self):
        # batch_size = self.real_A.shape[0]
        # self.t = torch.randint(0, 100, (batch_size,), device=self.device).long()
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.noise_real_A = torch.randn_like(self.real_A)
        self.real_A_noise = self.diffusion.q_sample(self.real_A, self.t, noise=self.noise_real_A)
        # self.noise_real_A = torch.randn_like(self.real_A[:,:1,...])
        # self.real_A_noise = torch.cat([self.diffusion.q_sample(self.real_A[:,:1,...], self.t_cup, noise=self.noise_real_A),
        #                                 self.diffusion.q_sample(self.real_A[:,1:,...], self.t_disc, noise=self.noise_real_A)], dim=1)
        # self.real_A_latent = self.netDenoise_A(self.real_A_noise, self.t)
        # self.fake_B = self.netG_A(torch.cat([self.real_A_noise, self.real_A_latent], dim=1))  # G_A(A)
        self.fake_B = self.netG_A(self.real_A_noise)  # G_A(A)

        self.noise_fake_B = torch.randn_like(self.fake_B)
        self.fake_B_noise = self.diffusion.q_sample(self.fake_B, self.t, noise=self.noise_fake_B)
        # self.noise_fake_B = torch.randn_like(self.fake_B[:,:1,...])
        # self.fake_B_noise = torch.cat([self.diffusion.q_sample(self.fake_B[:,:1,...], self.t_cup, noise=self.noise_fake_B),
        #                                 self.diffusion.q_sample(self.fake_B[:,1:,...], self.t_disc, noise=self.noise_fake_B)], dim=1)
        # self.fake_B_latent = self.netDenoise_B(self.fake_B_noise, self.t)
        # self.rec_A = self.netG_B(torch.cat([self.fake_B_noise, self.fake_B_latent], dim=1))   # G_B(G_A(A))
        self.rec_A = self.netG_B(self.fake_B_noise)   # G_B(G_A(A))

        self.noise_real_B = torch.randn_like(self.real_B)
        self.real_B_noise = self.diffusion.q_sample(self.real_B, self.t, noise=self.noise_real_B)
        # self.noise_real_B = torch.randn_like(self.real_B[:,:1,...])
        # self.real_B_noise = torch.cat([self.diffusion.q_sample(self.real_B[:,:1,...], self.t_cup, noise=self.noise_real_B),
        #                                 self.diffusion.q_sample(self.real_B[:,1:,...], self.t_disc, noise=self.noise_real_B)], dim=1)
        # self.real_B_latent = self.netDenoise_B(self.real_B_noise, self.t)
        # self.fake_A = self.netG_B(torch.cat([self.real_B_noise, self.real_B_latent], dim=1))  # G_B(B)
        self.fake_A = self.netG_B(self.real_B_noise)  # G_B(B)

        self.noise_fake_A = torch.randn_like(self.fake_A)
        self.fake_A_noise = self.diffusion.q_sample(self.fake_A, self.t, noise=self.noise_fake_A)
        # self.noise_fake_A = torch.randn_like(self.fake_A[:,:1,...])
        # self.fake_A_noise = torch.cat([self.diffusion.q_sample(self.fake_A[:,:1,...], self.t_cup, noise=self.noise_fake_A),
        #                                 self.diffusion.q_sample(self.fake_A[:,1:,...], self.t_disc, noise=self.noise_fake_A)], dim=1)
        # self.fake_A_latent = self.netDenoise_A(self.fake_A_noise, self.t)
        # self.rec_B = self.netG_A(torch.cat([self.fake_A_noise, self.fake_A_latent], dim=1))   # G_A(G_B(B))
        self.rec_B = self.netG_A(self.fake_A_noise)   # G_A(G_B(B))
        

    def get_output_B(self, input, type1='one', type2='one'):
        # # t = random.randint(200, 500)
        # batch_size = input.shape[0]
        # t = torch.randint(0, 100, (batch_size,), device=self.device).long()
        # logits = self.netG_N(input)
        # y = get_rep_outputs(logits, 0.5, True)
        # column_vector = torch.arange(self.opt.shift+1, self.opt.shift+self.opt.max_timestep+1).view(self.opt.max_timestep, 1).cuda()
        # t = y @ column_vector.float()
        logits1, logits2, logits3 = self.netG_N(input)
        y1 = get_rep_outputs(logits1, 0.5, True)
        y2 = get_rep_outputs(logits2, 0.5, True)
        y3 = get_rep_outputs(logits3, 0.5, True)
        column_vector1 = torch.arange(0, self.opt.max_timestep//100).view(self.opt.max_timestep//100, 1).cuda()
        column_vector2 = torch.arange(0, 10).view(10, 1).cuda()
        column_vector3 = torch.arange(1, 10).view(9, 1).cuda()
        # t = (y1.permute(0,2,3,1) @ column_vector1.float()) * 100 + (y2.permute(0,2,3,1) @ column_vector2.float()) * 10 + (y3.permute(0,2,3,1) @ column_vector3.float())
        t = (y1 @ column_vector1.float()) * 100 + (y2 @ column_vector2.float()) * 10 + (y3 @ column_vector3.float())

        # logits1, logits2, logits3 = self.netG_N_disc(input)
        # y1 = get_rep_outputs(logits1, 0.5, True)
        # y2 = get_rep_outputs(logits2, 0.5, True)
        # y3 = get_rep_outputs(logits3, 0.5, True)
        # column_vector1 = torch.arange(0, self.opt.max_timestep//100).view(self.opt.max_timestep//100, 1).cuda()
        # column_vector2 = torch.arange(0, 10).view(10, 1).cuda()
        # column_vector3 = torch.arange(1, 10).view(9, 1).cuda()
        # t_disc = (y1 @ column_vector1.float()) * 100 + (y2 @ column_vector2.float()) * 10 + (y3 @ column_vector3.float())

        noise_input = torch.randn_like(input)
        input_noise = self.diffusion.q_sample(input, t, noise=noise_input)
        # noise_input = torch.randn_like(input[:,:1,...])
        # input_noise = torch.cat([self.diffusion.q_sample(input[:,:1,...], t_cup, noise=noise_input),
        #                             self.diffusion.q_sample(input[:,1:,...], t_disc, noise=noise_input)], dim=1)
        # input_latent = self.netDenoise_B(input_noise, torch.full((input.shape[0],), t, device=self.device, dtype=torch.long))
        if type1 == 'one':
            output1 = self.netG_B(input_noise)  # denoise one step
        else:
            output1 = self.diffusion.sample(self.netDenoise_B, img=input_noise, t=t)[-1] #denoise step by step
            output1 = torch.from_numpy(output1).to(self.device)
        
        noise_output1 = torch.randn_like(output1)
        output1_noise = self.diffusion.q_sample(output1, t, noise=noise_output1)
        # noise_output1 = torch.randn_like(output1[:,:1,...])
        # output1_noise = torch.cat([self.diffusion.q_sample(output1[:,:1,...], t_cup, noise=noise_output1),
        #                             self.diffusion.q_sample(output1[:,1:,...], t_disc, noise=noise_output1)], dim=1)
        # output1_latent = self.netDenoise_A(output1_noise, torch.full((input.shape[0],), t, device=self.device, dtype=torch.long))
        if type2 == 'one':
            output2 = self.netG_A(output1_noise)
        else:
            output2 = self.diffusion.sample(self.netDenoise_A, img=output1_noise, t=t)[-1]
            output2 = torch.from_numpy(output2).to(self.device)

        # return output1, output2, t  #refine, recon
        return output1, output2, input_noise, output1_noise, t

    # def compute_visuals(self):
    #     self.visual_names.extend(['fake_B_sample', 'rec_A_sample', 'fake_A_sample', 'rec_B_sample'])
    #     t = random.randint(200, 500)
    #     batch_size = self.real_A.shape[0]
    #     self.noise_real_A = torch.randn_like(self.real_A)
    #     self.real_A_noise = self.diffusion.q_sample(self.real_A, torch.full((batch_size,), t, device=self.device, dtype=torch.long), noise=self.noise_real_A)
    #     self.real_A_latent = self.netDenoise_A(self.real_A_noise, torch.full((batch_size,), t, device=self.device, dtype=torch.long))
    #     self.fake_B = self.netG_A(torch.cat([self.real_A_noise, self.real_A_latent], dim=1))  # G_A(A)
    #     self.fake_B_sample = self.diffusion.sample(self.netDenoise_A, img=self.real_A_noise, t=t)[-1] #denoise step by step
    #     self.fake_B_sample = torch.from_numpy(self.fake_B_sample).to(self.device)
        
    #     self.noise_fake_B = torch.randn_like(self.fake_B)
    #     self.fake_B_noise = self.diffusion.q_sample(self.fake_B, torch.full((batch_size,), t, device=self.device, dtype=torch.long), noise=self.noise_fake_B)
    #     self.fake_B_latent = self.netDenoise_B(self.fake_B_noise, torch.full((batch_size,), t, device=self.device, dtype=torch.long))
    #     self.rec_A = self.netG_B(torch.cat([self.fake_B_noise, self.fake_B_latent], dim=1))   # G_B(G_A(A))
    #     self.rec_A_sample = self.diffusion.sample(self.netDenoise_B, img=self.fake_B_noise, t=t)[-1] #denoise step by step
    #     self.rec_A_sample = torch.from_numpy(self.rec_A_sample).to(self.device)
        
    #     self.noise_real_B = torch.randn_like(self.real_B)
    #     self.real_B_noise = self.diffusion.q_sample(self.real_B, torch.full((batch_size,), t, device=self.device, dtype=torch.long), noise=self.noise_real_B)
    #     self.real_B_latent = self.netDenoise_B(self.real_B_noise, torch.full((batch_size,), t, device=self.device, dtype=torch.long))
    #     self.fake_A = self.netG_B(torch.cat([self.real_B_noise, self.real_B_latent], dim=1))  # G_B(B)
    #     self.fake_A_sample = self.diffusion.sample(self.netDenoise_B, img=self.real_B_noise, t=t)[-1] #denoise step by step
    #     self.fake_A_sample = torch.from_numpy(self.fake_A_sample).to(self.device)
        
    #     self.noise_fake_A = torch.randn_like(self.fake_A)
    #     self.fake_A_noise = self.diffusion.q_sample(self.fake_A, torch.full((batch_size,), t, device=self.device, dtype=torch.long), noise=self.noise_fake_A)
    #     self.fake_A_latent = self.netDenoise_A(self.fake_A_noise, torch.full((batch_size,), t, device=self.device, dtype=torch.long))
    #     self.rec_B = self.netG_A(torch.cat([self.fake_A_noise, self.fake_A_latent], dim=1))   # G_A(G_B(B))
    #     self.rec_B_sample = self.diffusion.sample(self.netDenoise_A, img=self.fake_A_noise, t=t)[-1] #denoise step by step
    #     self.rec_B_sample = torch.from_numpy(self.rec_B_sample).to(self.device)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_N = self.opt.lambda_N
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            # self.noise_real_B = torch.randn_like(self.real_B)
            # self.real_B_noise = self.diffusion.q_sample(self.real_B, self.t, noise=self.noise_real_B)
            # self.real_B_latent = self.netDenoise_B(self.real_B_noise, self.t)
            self.idt_A = self.netG_A(self.real_B_noise)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            # self.noise_real_A = torch.randn_like(self.real_A)
            # self.real_A_noise = self.diffusion.q_sample(self.real_A, self.t, noise=self.noise_real_A)
            # self.real_A_latent = self.netDenoise_A(self.real_A_noise, self.t)
            self.idt_B = self.netG_B(self.real_A_noise)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # #diff loss
        # self.loss_diff_real_A = F.mse_loss(self.real_A_latent, self.real_A_noise) 
        # self.loss_diff_fake_B = F.mse_loss(self.fake_B_latent, self.fake_B_noise) 
        # self.loss_diff_real_B = F.mse_loss(self.real_B_latent, self.real_B_noise) 
        # self.loss_diff_fake_A = F.mse_loss(self.fake_A_latent, self.fake_A_noise) 
        # self.loss_diff = (self.loss_diff_real_A + self.loss_diff_fake_B + self.loss_diff_real_B + self.loss_diff_fake_A) * lambda_N
        # combined loss and calculate gradients
        # mask = torch.zeros_like(self.rec_B)
        # mask[self.rec_B>=0.75] = 1
        # self.loss_circularity = circularity_2label(mask * self.rec_B)
        self.loss_circularity = 0
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_circularity
        self.loss_G.backward()
        
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
