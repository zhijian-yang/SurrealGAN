import os
import time
from typing import Any, TextIO

import numpy as np
import pandas
import pandas as pd
from torch.autograd import Variable
from tqdm import tqdm

from .model import SurrealGAN, dotdict
from .utils import check_multimodel_agreement, parse_train_data, parse_validation_data

__author__ = "Zhijian Yang"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Zhijian Yang"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Zhijian Yang"
__email__ = "zhijianyang@outlook.com"
__status__ = "Development"


class Surreal_GAN_train:

    def __init__(
        self,
        npattern: int,
        final_saving_epoch: int,
        recons_loss_threshold: float,
        mono_loss_threshold: float,
        lam: float = 0.2,
        zeta: float = 80,
        kappa: float = 80,
        gamma: float = 6,
        mu: float = 500,
        eta: float = 6,
        alpha: float = 0.05,
        batchsize: float = 25,
        lipschitz_k: float = 0.5,
        beta1: float = 0.5,
        lr: float = 0.0002,
        max_gnorm: float = 100,
        eval_freq: float = 100,
        print_freq: int = 1000,
        saving_freq: int = 1000,
        early_stop_thresh: float = 0.02,
    ):
        self.opt = dotdict({})
        self.opt.npattern = npattern
        self.opt.final_saving_epoch = final_saving_epoch
        self.opt.recons_loss_threshold = recons_loss_threshold
        self.opt.mono_loss_threshold = mono_loss_threshold
        self.opt.lam = lam
        self.opt.mu = mu
        self.opt.zeta = zeta
        self.opt.kappa = kappa
        self.opt.gamma = gamma
        self.opt.eta = eta
        self.opt.alpha = alpha
        self.opt.batchsize = batchsize
        self.opt.lipschitz_k = lipschitz_k
        self.opt.beta1 = beta1
        self.opt.lr = lr
        self.opt.max_gnorm = max_gnorm
        self.opt.print_freq = print_freq
        self.opt.eval_freq = eval_freq
        self.opt.saving_freq = saving_freq
        self.opt.early_stop_thresh = early_stop_thresh

    @staticmethod
    def print_log(result_f: TextIO, message: str) -> None:
        result_f.write(message + "\n")
        result_f.flush()
        print(message)

    @staticmethod
    def format_log(
        epoch: int, epoch_iter: int, measures: dict, t: float, prefix: bool = True
    ) -> str:
        message = "(epoch: %d, iters: %d, time: %.3f) " % (epoch, epoch_iter, t)
        if not prefix:
            message = " " * len(message)
        for key, value in measures.items():
            message += "%s: %.4f " % (key, value)
        return message

    def parse_data(
        self,
        data: pandas.DataFrame,
        covariate: Any,
        random_seed: float,
        data_fraction: int,
    ) -> Any:
        (
            cn_train_dataset,
            pt_train_dataset,
            correction_variables,
            normalization_variables,
        ) = parse_train_data(  # type: ignore
            data, covariate, random_seed, data_fraction, self.opt.batchsize  # type: ignore
        )
        cn_eval_dataset, pt_eval_dataset = parse_validation_data(  # type: ignore
            data, covariate, correction_variables, normalization_variables
        )
        self.opt.nROI = pt_eval_dataset.shape[1]
        self.opt.n_val_data = pt_eval_dataset.shape[0]
        return (
            cn_train_dataset,
            pt_train_dataset,
            cn_eval_dataset,
            pt_eval_dataset,
            correction_variables,
            normalization_variables,
        )

    def train(
        self,
        data: pandas.DataFrame,
        covariate: Any,
        save_dir: str,
        repetition: int,
        random_seed: float = 0.0,
        data_fraction: int = 1,
        verbose: bool = True,
    ) -> bool:
        if verbose:
            result_f = open("%s/results.txt" % save_dir, "w")

        (
            cn_train_dataset,
            pt_train_dataset,
            eval_X,
            eval_Y,
            correction_variables,
            normalization_variables,
        ) = self.parse_data(data, covariate, random_seed, data_fraction)
        self.opt.correction_variables = correction_variables
        self.opt.normalization_variables = normalization_variables

        # create_model
        model = SurrealGAN()
        model.create(self.opt)

        total_steps = 0
        print_start_time = time.time()
        # best_agreement = 0
        # best_epoch = 0
        # best_dimension_corr, best_difference_corr = [], []

        criterion_loss_list = [
            [0 for _ in range(2)] for _ in range(3)
        ]  # number of consecutive epochs with aq and cluster_loss < threshold
        # predicted_label_past = np.zeros(self.opt.n_val_data)

        if self.opt.final_saving_epoch % self.opt.saving_freq == 0:  # type: ignore
            save_epoch = [
                i * self.opt.saving_freq  # type: ignore
                for i in range(
                    2, self.opt.final_saving_epoch // self.opt.saving_freq + 1  # type: ignore
                )
            ]
        else:
            save_epoch = [
                i * self.opt.saving_freq  # type: ignore
                for i in range(
                    2, self.opt.final_saving_epoch // self.opt.saving_freq + 1  # type: ignore
                )
            ] + [
                self.opt.final_saving_epoch  # type: ignore
            ]
        save_epoch_index = 0
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not verbose:
            pbar = tqdm(total=self.opt.final_saving_epoch + 2000)  # type: ignore
        for epoch in range(1, self.opt.final_saving_epoch + 2001):  # type: ignore
            if not verbose:
                pbar.update(1)
            # epoch_start_time = time.time()
            epoch_iter = 0
            for i, pt_data in enumerate(pt_train_dataset):
                cn_data = cn_train_dataset.next()
                real_X, real_Y = Variable(cn_data["x"]), Variable(pt_data["y"])

                total_steps += self.opt.batchsize  # type: ignore
                epoch_iter += self.opt.batchsize  # type: ignore

                losses = model.train_instance(real_X, real_Y, self.opt.alpha)  # type: ignore

                if total_steps % self.opt.print_freq == 0:  # type: ignore
                    t = (time.time() - print_start_time) / self.opt.batchsize  # type: ignore
                    if verbose:
                        self.print_log(
                            result_f, self.format_log(epoch, epoch_iter, losses, t)
                        )
                        print_start_time = time.time()

            if epoch % self.opt.eval_freq == 0:  # type: ignore
                t = time.time()

                loss_names = ["loss_recons", "loss_mono"]
                for _ in range(2):
                    criterion_loss_list[_].append(losses[loss_names[_]])

                for _ in range(2):
                    criterion_loss_list[_].pop(0)

                t = time.time() - t
                res_str_list = ["[%d], TIME: %.4f" % (epoch, t)]

                if (
                    max(criterion_loss_list[0]) < self.opt.recons_loss_threshold  # type: ignore
                    and max(criterion_loss_list[1]) < self.opt.mono_loss_threshold  # type: ignore
                    and epoch > save_epoch[save_epoch_index]
                ):
                    model.save(
                        save_dir,
                        save_epoch[save_epoch_index],
                        "model" + str(random_seed),
                    )
                    res_str_list += ["*** Saving Criterion Satisfied ***"]
                    res_str = "\n".join(["-" * 60] + res_str_list + ["-" * 60])
                    agreement_list = check_multimodel_agreement(  # type: ignore
                        data,
                        covariate,
                        save_dir,
                        save_epoch[save_epoch_index],
                        repetition,
                        self.opt.npattern,  # type: ignore
                    )
                    if os.path.exists(os.path.join(save_dir, "model_agreements.csv")):
                        agreement_f = pd.read_csv(
                            os.path.join(save_dir, "model_agreements.csv")
                        )
                    else:
                        agreement_f = pd.DataFrame(
                            columns=[
                                "epoch",
                                "Rindices_corr",
                                "dimension_corr",
                                "difference_corr",
                                "best_Rindices_corr",
                                "best_dimension_corr",
                                "best_difference_corr",
                                "best_model",
                                "stop",
                            ]
                        )
                    if len(agreement_list) > 0:
                        dimension_corr, difference_corr = (
                            agreement_list[1],
                            agreement_list[0],
                        )
                        Rindices_corr = [
                            (a + b) / 2 for a, b in zip(dimension_corr, difference_corr)
                        ]
                        best_model = Rindices_corr.index(max(Rindices_corr))
                        if (
                            agreement_f.iloc[:-2]["Rindices_corr"].max()
                            - self.opt.early_stop_thresh  # type: ignore
                        ) > max(
                            agreement_f.iloc[-2:]["Rindices_corr"].max(),
                            (np.mean(dimension_corr) + np.mean(difference_corr)) / 2,
                        ):
                            stop = "yes"
                        else:
                            stop = "no"
                        if save_epoch[save_epoch_index] not in agreement_f["epoch"]:
                            agreement_f.loc[len(agreement_f)] = {
                                "epoch": save_epoch[save_epoch_index],
                                "Rindices_corr": (
                                    np.mean(dimension_corr) + np.mean(difference_corr)
                                )
                                / 2,
                                "dimension_corr": np.mean(dimension_corr),
                                "difference_corr": np.mean(difference_corr),
                                "best_Rindices_corr": Rindices_corr[best_model],
                                "best_dimension_corr": dimension_corr[best_model],
                                "best_difference_corr": difference_corr[best_model],
                                "best_model": best_model,
                                "stop": stop,
                            }
                            agreement_f.to_csv(
                                os.path.join(save_dir, "model_agreements.csv"),
                                index=False,
                            )

                    if (agreement_f["stop"] == "yes").any():
                        return True
                    save_epoch_index += 1

                    if verbose:
                        self.print_log(result_f, res_str)
                    if save_epoch_index == len(save_epoch):
                        if not verbose:
                            pbar.close()
                        if verbose:
                            result_f.close()
                        return True

                res_str = "\n".join(["-" * 60] + res_str_list + ["-" * 60])
                res_str_list += [
                    "*** Max epoch reached and criterion not satisfied ***"
                ]
                if verbose:
                    self.print_log(result_f, res_str)
        agreement_f.close()
        if verbose:
            result_f.close()
        if not verbose:
            pbar.close()
        return False
