import os
from collections import OrderedDict
from itertools import chain as ichain
from typing import Any, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.kl import kl_divergence

from .copula import construct_scale_tril, guassian_colula_distribution
from .networks import (
    define_Linear_Decomposer,
    define_Linear_Discriminator,
    define_Linear_Mapping,
    define_Linear_Reconstruction,
)

__author__ = "Zhijian Yang"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Zhijian Yang"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Zhijian Yang"
__email__ = "zhijianyang@outlook.com"
__status__ = "Development"


# sample from discrete uniform random variable and construct SUB variable.
def sample_z_previous(real_X: Any, index: int, npattern: int) -> Any:
    Tensor = torch.FloatTensor
    z_idx = torch.empty(real_X.size(0), dtype=torch.long)
    z_idx[:] = index
    z = Tensor(real_X.size(0), npattern).fill_(0)
    zt_random = Variable(
        real_X.data.new(real_X.size(0), 1).uniform_(0.00000001, 0.99999999)
    )
    z = z.scatter(1, z_idx.unsqueeze(1), zt_random)
    # zt_random = zt_random.scatter_(1, index.unsqueeze(1), 0)
    return z


def sample_z_later(
    pre: np.ndarray, real_X: Any, index: int, npattern: int
) -> torch.FloatTensor:
    Tensor = torch.FloatTensor
    pre = pre[:, index : index + 1]
    z_idx = torch.empty(real_X.size(0), dtype=torch.long)
    z_idx[:] = index
    z = Tensor(real_X.size(0), npattern).fill_(0)
    zt_random = Variable((1 - pre) * torch.rand(real_X.size(0), 1) + pre)
    z = z.scatter(1, z_idx.unsqueeze(1), zt_random)
    return z


def sample_z_cn(real_X: Any, npattern: int) -> Variable:
    z_random = Variable(
        real_X.data.new(real_X.size(0), npattern).uniform_(0.000000001, 0.05)
    )
    return z_random


def criterion_GAN(pred: Any, target_is_real: bool, prob: float) -> Any:
    if target_is_real:
        target_var = Variable(pred.data.new(pred.shape[0]).long().fill_(0.0))
        loss = (F.cross_entropy(pred, target_var, reduce=False) * prob).mean()
    else:
        target_var = Variable(pred.data.new(pred.shape[0]).long().fill_(1.0))
        loss = (F.cross_entropy(pred, target_var, reduce=False) * prob).mean()
    return loss


def criterion_orthogonal(change_batch_sum: torch.Tensor, npattern: int) -> torch.Tensor:
    change_batch_sum = torch.abs(change_batch_sum)
    change_batch_sum = change_batch_sum / (
        torch.norm(change_batch_sum, dim=1).view(npattern, 1)
    )
    matrix = torch.matmul(change_batch_sum, torch.transpose(change_batch_sum, 0, 1))
    return F.mse_loss(matrix, torch.eye(npattern))


def mono_loss(later: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(
        torch.max(torch.abs(prev) - torch.abs(later), torch.zeros_like(later)),
        torch.zeros_like(later),
    )


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get

    def __setattr__(self: "dotdict", name: str, value: Any) -> None:
        self.__setitem__(name, value)

    # Explicitly define __delattr__ type
    def __delattr__(self: "dotdict", name: str) -> None:
        self.__delitem__(name)


class SurrealGAN(object):
    def __init__(self) -> None:
        self.phi = None
        self.opt: Union[dotdict, None] = None

        # definition of all netwotks
        self.netMapping = None
        self.netReconstruction = None
        self.netDiscriminator = None

        # definition of all optimizers
        self.optimizer_M = None
        self.optimizer_D = None

        # definition of all criterions
        self.criterionGAN = criterion_GAN
        self.criterionChange = F.l1_loss
        self.criterionMono = mono_loss
        self.criterionRecons = F.mse_loss
        self.criterionOrtho = criterion_orthogonal

    def create(self, opt: dotdict) -> None:

        self.opt = opt

        # definition of all netwotks
        self.netMapping = define_Linear_Mapping(self.opt.nROI, self.opt.npattern)  # type: ignore
        self.netReconstruction = define_Linear_Reconstruction(
            self.opt.nROI, self.opt.npattern  # type: ignore
        )
        self.netDiscriminator = define_Linear_Discriminator(self.opt.nROI)  # type: ignore
        self.netDecomposer = define_Linear_Decomposer(self.opt.nROI, self.opt.npattern)  # type: ignore
        tril_indices = torch.tril_indices(
            row=self.opt.npattern, col=self.opt.npattern, offset=0  # type: ignore
        )
        self.phi = torch.nn.Parameter(
            torch.eye(self.opt.npattern)[tril_indices[0], tril_indices[1]]  # type: ignore
        )
        self.initial_latent_distribution = guassian_colula_distribution(
            torch.eye(self.opt.npattern), self.opt.npattern  # type: ignore
        )

        # definition of all optimizers
        self.optimizer_M = torch.optim.Adam(
            ichain(
                self.netMapping.parameters(),  # type: ignore
                self.netDecomposer.parameters(),
                self.netReconstruction.parameters(),  # type: ignore
            ),
            lr=self.opt.lr,  # type: ignore
            betas=(self.opt.beta1, 0.999),  # type: ignore
            weight_decay=2.5 * 1e-3,
        )
        self.optimizer_D = torch.optim.Adam(
            ichain(self.netDiscriminator.parameters()),  # type: ignore
            lr=self.opt.lr / 5.0,  # type: ignore
            betas=(self.opt.beta1, 0.999),  # type: ignore
        )
        self.optimizer_phi = torch.optim.Adam(
            iter([self.phi]), lr=self.opt.lr / 30.0, betas=(self.opt.beta1, 0.999)  # type: ignore
        )

    def train_instance(self, x: Any, real_y: Any, alpha: float) -> OrderedDict:
        z_pre = [
            sample_z_previous(x, i, self.opt.npattern) for i in range(self.opt.npattern)  # type: ignore
        ]
        # z_pre_sum = torch.sum(torch.stack(z_pre), dim=0)
        z_later = [
            sample_z_later(z_pre[i], x, i, self.opt.npattern)  # type: ignore
            for i in range(self.opt.npattern)  # type: ignore
        ]
        # z_later_sum = torch.sum(torch.stack(z_later), dim=0)

        fake_y_pre_change = [
            self.netMapping.forward(x, z_pre[i]) for i in range(self.opt.npattern)  # type: ignore
        ]
        fake_y_later_change = [
            self.netMapping.forward(x, z_later[i]) for i in range(self.opt.npattern)  # type: ignore
        ]

        fake_y_pre_change_tensor = torch.stack(fake_y_pre_change)
        fake_y_later_change_tensor = torch.stack(fake_y_later_change)

        fake_y_pre = x + torch.sum(fake_y_pre_change_tensor, dim=0)
        # fake_y_later = x + torch.sum(fake_y_later_change_tensor, dim=0)

        fake_y_change_batch_sum = torch.mean(
            torch.cat((fake_y_pre_change_tensor, fake_y_later_change_tensor), 1), dim=1
        )

        z_cn = sample_z_cn(x, self.opt.npattern)  # type: ignore
        fake_y_cn = self.netMapping.forward(x, z_cn) + x  # type: ignore

        fake_y = fake_y_pre
        fake_change_decompose = fake_y_pre_change_tensor
        fake_z_decompose = torch.stack(z_pre)

        latent_distribution = guassian_colula_distribution(
            construct_scale_tril(self.phi, self.opt.npattern), self.opt.npattern  # type: ignore
        )
        post_prob = torch.exp(
            latent_distribution.log_prob(torch.sum(fake_z_decompose, dim=0))
        )

        # Discriminator loss
        pred_fake_y = self.netDiscriminator.forward(fake_y.detach())  # type: ignore
        loss_D_fake_y = self.criterionGAN(pred_fake_y, False, post_prob.detach())
        pred_true_y = self.netDiscriminator.forward(real_y)  # type: ignore
        loss_D_true_y = self.criterionGAN(pred_true_y, True, post_prob.detach())
        loss_D = 0.5 * (loss_D_fake_y + loss_D_true_y)

        # update weights of discriminator
        self.optimizer_D.zero_grad()  # type: ignore
        loss_D.backward()
        # gnorm_D = torch.nn.utils.clip_grad_norm_(
        #     self.netDiscriminator.parameters(), self.opt.max_gnorm
        # )
        self.optimizer_D.step()  # type: ignore

        # Phi loss
        loss_phi = self.criterionGAN(
            self.netDiscriminator.forward(fake_y.detach()).detach(), True, post_prob  # type: ignore
        ) + alpha * kl_divergence(self.initial_latent_distribution, latent_distribution)
        self.optimizer_phi.zero_grad()
        loss_phi.backward()
        self.optimizer_phi.step()

        # Mapping related loss
        pred_fake_y = self.netDiscriminator.forward(fake_y)  # type: ignore
        loss_mapping = self.criterionGAN(pred_fake_y, True, post_prob.detach())

        decomposed_fake_y = self.netDecomposer.forward(fake_y)

        reconst_zt = [
            self.netReconstruction.forward(decomposed_fake_y[i])  # type: ignore
            for i in range(self.opt.npattern)  # type: ignore
        ]

        recons_loss = self.criterionRecons(torch.stack(reconst_zt), fake_z_decompose)

        decompose_loss = self.criterionRecons(
            torch.stack(decomposed_fake_y), fake_change_decompose
        )

        change_loss = self.criterionChange(fake_y, x)

        mono_loss = self.criterionMono(
            fake_y_later_change_tensor, fake_y_pre_change_tensor
        )

        orthogonal_loss = self.criterionOrtho(
            fake_y_change_batch_sum, self.opt.npattern  # type: ignore
        )

        cn_loss = self.criterionChange(fake_y_cn, x)

        loss_G = (
            (1 / torch.mean(post_prob.detach())) * loss_mapping
            + self.opt.lam * orthogonal_loss  # type: ignore
            + self.opt.zeta * recons_loss  # type: ignore
            + self.opt.kappa * decompose_loss  # type: ignore
            + self.opt.gamma * change_loss  # type: ignore
            + self.opt.mu * mono_loss  # type: ignore
            + self.opt.eta * cn_loss  # type: ignore
        )  # 7

        # update weights of mapping/reconstruction/decomposer function
        self.optimizer_M.zero_grad()  # type: ignore
        loss_G.backward()
        # gnorm_M = torch.nn.utils.clip_grad_norm_(
        #     self.netMapping.parameters(), self.opt.max_gnorm
        # )
        self.optimizer_M.step()  # type: ignore

        # perform weight clipping
        for p in self.netMapping.parameters():  # type: ignore
            p.data.clamp_(-self.opt.lipschitz_k, self.opt.lipschitz_k)  # type: ignore
        for p in self.netReconstruction.parameters():  # type: ignore
            p.data.clamp_(-self.opt.lipschitz_k, self.opt.lipschitz_k)  # type: ignore
        for p in self.netDecomposer.parameters():
            p.data.clamp_(-self.opt.lipschitz_k, self.opt.lipschitz_k)  # type: ignore

        # Return dicts
        losses = OrderedDict(
            [
                ("Discriminator_loss", loss_D.item()),
                ("Mapping_loss", loss_mapping.item()),
                ("loss_change", change_loss.item()),
                ("loss_orthogonal", orthogonal_loss.item()),
                ("loss_recons", recons_loss.item()),
                ("loss_mono", 100 * mono_loss.item()),
                ("loss_cn", cn_loss.item()),
                ("loss_decompose", decompose_loss.item()),
            ]
        )

        return losses

    # return decomposed changes of each dimension given PT data
    def decompose(self, real_y: Any) -> List[np.ndarray]:
        decompose_y = self.netDecomposer.forward(real_y)
        return [decompose_y[i].detach().numpy() for i in range(self.opt.npattern)]  # type: ignore

    # return rindices given PT data
    def predict_rindices(self, real_y: Any) -> np.ndarray:
        decompose_y = self.netDecomposer.forward(real_y)
        reconst_zt_decompose = [
            self.netReconstruction.forward(decompose_y[i])  # type: ignore
            for i in range(self.opt.npattern)  # type: ignore
        ]
        t = torch.sum(torch.stack(reconst_zt_decompose), dim=0)
        return t.detach().numpy()

    # return generated patient data with given latent variable
    def predict_Y(self, x: Any, z: Any) -> np.ndarray:
        return self.netMapping.forward(x, z).detach().numpy()  # type: ignore

    def get_corr(self):  # type: ignore
        # scale_tril = construct_scale_tril(self.phi, self.opt.npattern)
        return

    # save checkpoint
    def save(self, save_dir: str, epoch: int, chk_name: str) -> None:
        chk_path = os.path.join(save_dir, chk_name)
        if os.path.exists(chk_path):
            checkpoint = torch.load(chk_path)
        else:
            checkpoint = {}
            checkpoint.update(self.opt)
        checkpoint[epoch] = {
            "netMapping": self.netMapping.state_dict(),  # type: ignore
            "netDiscriminator": self.netDiscriminator.state_dict(),  # type: ignore
            "optimizer_D": self.optimizer_D.state_dict(),  # type: ignore
            "optimizer_M": self.optimizer_M.state_dict(),  # type: ignore
            "optimizer_phi": self.optimizer_phi.state_dict(),
            "netReconstruction": self.netReconstruction.state_dict(),  # type: ignore
            "netDecomposer": self.netDecomposer.state_dict(),
            "phi": self.phi,
        }
        for i in range(100000):
            try:
                torch.save(checkpoint, chk_path)
            except Exception:
                continue
            else:
                break

    def load_opt(self, checkpoint: OrderedDict) -> None:
        self.opt = dotdict({})
        for key in checkpoint.keys():
            if not isinstance(key, int):
                self.opt[key] = checkpoint[key]

    # load trained model
    def load(self, chk_path: str, epoch: int) -> None:
        for i in range(100000):
            try:
                checkpoint_all_epoch = torch.load(chk_path)
            except Exception:
                continue
            else:
                break
        checkpoint = checkpoint_all_epoch[epoch]
        self.load_opt(checkpoint_all_epoch)
        # definition of all netwotks
        self.netMapping = define_Linear_Mapping(self.opt.nROI, self.opt.npattern)  # type: ignore
        self.netReconstruction = define_Linear_Reconstruction(  # type: ignore
            self.opt.nROI, self.opt.npattern  # type: ignore
        )
        self.netDiscriminator = define_Linear_Discriminator(self.opt.nROI)  # type: ignore
        self.netDecomposer = define_Linear_Decomposer(self.opt.nROI, self.opt.npattern)  # type: ignore
        tril_indices = torch.tril_indices(
            row=self.opt.npattern, col=self.opt.npattern, offset=0  # type: ignore
        )
        self.phi = torch.nn.Parameter(
            torch.eye(self.opt.npattern)[tril_indices[0], tril_indices[1]]  # type: ignore
        )

        # definition of all optimizers
        self.optimizer_M = torch.optim.Adam(
            ichain(
                self.netMapping.parameters(),  # type: ignore
                self.netDecomposer.parameters(),
                self.netReconstruction.parameters(),  # type: ignore
            ),
            lr=self.opt.lr,  # type: ignore
            betas=(self.opt.beta1, 0.999),  # type: ignore
            weight_decay=2.5 * 1e-3,
        )
        self.optimizer_D = torch.optim.Adam(
            ichain(self.netDiscriminator.parameters()),  # type: ignore
            lr=self.opt.lr / 5.0,  # type: ignore
            betas=(self.opt.beta1, 0.999),  # type: ignore
        )
        self.optimizer_phi = torch.optim.Adam(
            iter([self.phi]), lr=self.opt.lr / 30.0, betas=(self.opt.beta1, 0.999)  # type: ignore
        )

        self.netMapping.load_state_dict(checkpoint["netMapping"])  # type: ignore
        self.netDiscriminator.load_state_dict(checkpoint["netDiscriminator"])  # type: ignore
        self.netReconstruction.load_state_dict(checkpoint["netReconstruction"])  # type: ignore
        self.netDecomposer.load_state_dict(checkpoint["netDecomposer"])
        self.optimizer_D.load_state_dict(checkpoint["optimizer_D"])  # type: ignore
        self.optimizer_M.load_state_dict(checkpoint["optimizer_M"])  # type: ignore
        self.optimizer_phi.load_state_dict(checkpoint["optimizer_phi"])
        self.phi = checkpoint["phi"]
