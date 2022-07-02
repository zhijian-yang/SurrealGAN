import os
import torch
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable
from itertools import chain as ichain
from .networks import define_Linear_Mapping, define_Linear_Reconstruction, define_Linear_Discriminator, define_Linear_Decomposer

__author__ = "Zhijian Yang"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Zhijian Yang"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Zhijian Yang"
__email__ = "zhijianyang@outlook.com"
__status__ = "Development"


#####sample from discrete uniform random variable and construct SUB variable. 
def sample_z_previous(real_X,index, npattern):
    Tensor = torch.FloatTensor
    z_idx = torch.empty(real_X.size(0), dtype=torch.long)
    z_idx[:] = index
    z = Tensor(real_X.size(0), npattern).fill_(0)
    zt_random = Variable(real_X.data.new(real_X.size(0),1).uniform_(0,1))
    z = z.scatter(1, z_idx.unsqueeze(1), zt_random)
    #zt_random = zt_random.scatter_(1, index.unsqueeze(1), 0)
    return z

def sample_z_later(pre,real_X, index,npattern):
    Tensor = torch.FloatTensor
    pre = pre[:,index:index+1]
    z_idx = torch.empty(real_X.size(0), dtype=torch.long)
    z_idx[:] = index
    z = Tensor(real_X.size(0), npattern).fill_(0)
    zt_random = Variable((1 - pre) * torch.rand(real_X.size(0),1) + pre)
    z = z.scatter(1, z_idx.unsqueeze(1), zt_random)
    return z

def sample_z_cn(real_X, npattern):
    z_random = Variable(real_X.data.new(real_X.size(0),npattern).uniform_(0,0.05))
    return z_random


def criterion_GAN(pred, target_is_real):
    if target_is_real:
        target_var = Variable(pred.data.new(pred.shape[0]).long().fill_(0.))
        loss=F.cross_entropy(pred, target_var)
    else:
        target_var = Variable(pred.data.new(pred.shape[0]).long().fill_(1.))
        loss = F.cross_entropy(pred, target_var)
    return loss

def criterion_orthogonal(change_batch_sum, npattern):
    change_batch_sum = torch.abs(change_batch_sum)
    change_batch_sum = change_batch_sum/(torch.norm(change_batch_sum,dim=1).view(npattern,1))
    matrix = torch.matmul(change_batch_sum, torch.transpose(change_batch_sum,0,1))
    return F.mse_loss(matrix,torch.eye(npattern))

def mono_loss(later, prev):
    return F.mse_loss(torch.max(torch.abs(prev)-torch.abs(later),torch.zeros_like(later)),torch.zeros_like(later))

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class SurrealGAN(object):
    def __init__(self):
        self.opt = None

        ##### definition of all netwotks
        self.netMapping = None
        self.netReconstruction= None
        self.netDiscriminator = None

        ##### definition of all optimizers
        self.optimizer_M = None
        self.optimizer_D = None

        ##### definition of all criterions
        self.criterionGAN = criterion_GAN
        self.criterionChange = F.l1_loss
        self.criterionMono = mono_loss
        self.criterionRecons = F.mse_loss
        self.criterionOrtho = criterion_orthogonal


    def create(self, opt):

        self.opt = opt

        ## definition of all netwotks
        self.netMapping = define_Linear_Mapping(self.opt.nROI,self.opt.npattern)
        self.netReconstruction = define_Linear_Reconstruction(self.opt.nROI,self.opt.npattern)
        self.netDiscriminator = define_Linear_Discriminator(self.opt.nROI,self.opt.npattern)
        self.netDecomposer = define_Linear_Decomposer(self.opt.nROI,self.opt.npattern)

        ## definition of all optimizers
        self.optimizer_M = torch.optim.Adam(ichain(self.netMapping.parameters(),self.netDecomposer.parameters(),self.netReconstruction.parameters()),
                                        lr=self.opt.lr, betas=(self.opt.beta1, 0.999),weight_decay=2.5*1e-3)
        self.optimizer_D = torch.optim.Adam(ichain(self.netDiscriminator.parameters()),
                                            lr=self.opt.lr/5., betas=(self.opt.beta1, 0.999))


    def train_instance(self, x, real_y):
        z_pre = [sample_z_previous(x,i,self.opt.npattern) for i in range(self.opt.npattern)]
        z_pre_sum = torch.sum(torch.stack(z_pre),dim=0)
        z_later = [sample_z_later(z_pre[i],x,i,self.opt.npattern) for i in range(self.opt.npattern)]
        z_later_sum = torch.sum(torch.stack(z_later),dim=0)

        fake_y_pre_change = [self.netMapping.forward(x,z_pre[i]) for i in range(self.opt.npattern)]
        fake_y_later_change = [self.netMapping.forward(x,z_later[i]) for i in range(self.opt.npattern)]

        fake_y_pre_change_tensor = torch.stack(fake_y_pre_change)
        fake_y_later_change_tensor = torch.stack(fake_y_later_change)

        fake_y_pre = x + torch.sum(fake_y_pre_change_tensor, dim=0)
        fake_y_later = x + torch.sum(fake_y_later_change_tensor, dim=0)

        fake_y_change_batch_sum = torch.mean(torch.cat((fake_y_pre_change_tensor , fake_y_later_change_tensor),1),dim=1)
        

        z_cn = sample_z_cn(x, self.opt.npattern)
        fake_y_cn = self.netMapping.forward(x,z_cn) + x

        fake_y = fake_y_pre
        fake_change_decompose = fake_y_pre_change_tensor
        fake_z_decompose = torch.stack(z_pre)

        ## Discriminator loss
        pred_fake_y = self.netDiscriminator.forward(fake_y.detach())
        loss_D_fake_y = self.criterionGAN(pred_fake_y, False)
        pred_true_y = self.netDiscriminator.forward(real_y)
        loss_D_true_y = self.criterionGAN(pred_true_y, True)
        loss_D= 0.5* (loss_D_fake_y + loss_D_true_y)

        ## update weights of discriminator
        self.optimizer_D.zero_grad()
        loss_D.backward()
        gnorm_D = torch.nn.utils.clip_grad_norm_(self.netDiscriminator.parameters(), self.opt.max_gnorm)
        self.optimizer_D.step()

        ## Mapping related loss
        pred_fake_y = self.netDiscriminator.forward(fake_y)
        loss_mapping = self.criterionGAN(pred_fake_y, True)

        decomposed_fake_y = self.netDecomposer.forward(fake_y)

        reconst_zt = [self.netReconstruction.forward(decomposed_fake_y[i]) for i in range(self.opt.npattern)]

        recons_loss = self.criterionRecons(torch.stack(reconst_zt), fake_z_decompose)

        decompose_loss = self.criterionRecons(torch.stack(decomposed_fake_y), fake_change_decompose)

        change_loss= self.criterionChange(fake_y, x)

        mono_loss = self.criterionMono(fake_y_later_change_tensor, fake_y_pre_change_tensor)

        orthogonal_loss = self.criterionOrtho(fake_y_change_batch_sum,self.opt.npattern)

        cn_loss = self.criterionChange(fake_y_cn, x)

        loss_G = loss_mapping+self.opt.lam*orthogonal_loss+ self.opt.zeta*recons_loss +self.opt.kappa*decompose_loss + self.opt.gamma*change_loss + self.opt.mu*mono_loss + self.opt.eta*cn_loss #7


        ## update weights of mapping/reconstruction/decomposer function
        self.optimizer_M.zero_grad()
        loss_G.backward()
        gnorm_M = torch.nn.utils.clip_grad_norm_(self.netMapping.parameters(), self.opt.max_gnorm)
        self.optimizer_M.step()

        ## perform weight clipping

        for p in self.netMapping.parameters():
            p.data.clamp_(-self.opt.lipschitz_k, self.opt.lipschitz_k)
        for p in self.netReconstruction.parameters():
            p.data.clamp_(-self.opt.lipschitz_k, self.opt.lipschitz_k)
        for p in self.netDecomposer.parameters():
            p.data.clamp_(-self.opt.lipschitz_k, self.opt.lipschitz_k)


        ## Return dicts
        losses=OrderedDict([('Discriminator_loss', loss_D.item()),('Mapping_loss', loss_mapping.item()),('loss_change', change_loss.item()),('loss_orthogonal', orthogonal_loss.item())
            ,('loss_recons', recons_loss.item()),('loss_mono', 100*mono_loss.item()),('loss_cn', cn_loss.item()),('loss_decompose', decompose_loss.item())])

        
        return losses


    
    ## return decomposed changes of each dimension given PT data
    def decompose(self,real_y):
        decompose_y=self.netDecomposer.forward(real_y)
        return [decompose_y[i].detach().numpy() for i in range(self.opt.npattern)]

    ## return rindices given PT data
    def predict_rindices(self,real_y):
        decompose_y=self.netDecomposer.forward(real_y)
        reconst_zt_decompose = [self.netReconstruction.forward(decompose_y[i]) for i in range(self.opt.npattern)]
        t = torch.sum(torch.stack(reconst_zt_decompose),dim=0)
        return t.detach().numpy()

    ## return generated patient data with given zub variable
    def predict_Y(self, x, z):
        return self.netMapping.forward(x, z)

    ## save checkpoint    
    def save(self, save_dir, chk_name):
        chk_path = os.path.join(save_dir, chk_name)
        checkpoint = {
            'netMapping':self.netMapping.state_dict(),
            'netDiscriminator':self.netDiscriminator.state_dict(),
            'optimizer_D':self.optimizer_D.state_dict(),
            'optimizer_M':self.optimizer_M.state_dict(),
            'netReconstruction':self.netReconstruction.state_dict(),
            'netDecomposer':self.netDecomposer.state_dict(),
        }
        checkpoint.update(self.opt)
        torch.save(checkpoint, chk_path)

    def load_opt(self,checkpoint):
        self.opt = dotdict({})
        for key in checkpoint.keys():
            if key not in ['netMapping','netDiscriminator','netReconstruction','netDecomposer','optimizer_M','optimizer_D']:
                self.opt[key] = checkpoint[key]
        

    ## load trained model
    def load(self, chk_path):
        checkpoint = torch.load(chk_path)
        self.load_opt(checkpoint)
        ##### definition of all netwotks
        self.netMapping = define_Linear_Mapping(self.opt.nROI,self.opt.npattern)
        self.netReconstruction = define_Linear_Reconstruction(self.opt.nROI,self.opt.npattern)
        self.netDiscriminator = define_Linear_Discriminator(self.opt.nROI,self.opt.npattern)
        self.netDecomposer = define_Linear_Decomposer(self.opt.nROI,self.opt.npattern)

        ##### definition of all optimizers
        self.optimizer_M = torch.optim.Adam(ichain(self.netMapping.parameters(),self.netDecomposer.parameters(),self.netReconstruction.parameters()),
                                        lr=self.opt.lr, betas=(self.opt.beta1, 0.999),weight_decay=2.5*1e-3)
        self.optimizer_D = torch.optim.Adam(ichain(self.netDiscriminator.parameters()),
                                        lr=self.opt.lr/5., betas=(self.opt.beta1, 0.999))
        
        self.netMapping.load_state_dict(checkpoint['netMapping'])
        self.netDiscriminator.load_state_dict(checkpoint['netDiscriminator'])
        self.netReconstruction.load_state_dict(checkpoint['netReconstruction'])
        self.netDecomposer.load_state_dict(checkpoint['netDecomposer'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        self.optimizer_M.load_state_dict(checkpoint['optimizer_M'])
        self.load_opt(checkpoint)
            


        
