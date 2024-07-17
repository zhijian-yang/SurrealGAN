import os
from typing import Tuple

import numpy as np
import pandas
import pandas as pd

from .model import SurrealGAN
from .training import Surreal_GAN_train
from .utils import parse_validation_data

__author__ = "Zhijian Yang"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Zhijian Yang"]
__license__ = "See LICENSE file"
__version__ = "0.0.1"
__maintainer__ = "Zhijian Yang"
__email__ = "zhijianyang@outlook.com"
__status__ = "Development"


def representation_result(
    output_dir: str,
    data: pandas.DataFrame,
    final_saving_epoch: int,
    repetition: int,
    covariate: pandas.DataFrame = None,
) -> Tuple[np.ndarray, float, float, float, float, int, str]:
    """
    Function used for derive representation results from several saved models
    :param output_dir: str The directory where the model and results are saved
    :param data: Pandas DataFrame, The data used for testing the model, same format as training data
    :param final_saving_epoch: int, epoch number from which the last model will be saved and model training will be stopped if saving criteria satisfied
    :param repetition: int, number of repetition of training process
    :param covariate: Pandas DataFrame, The covariate data used for testing the model, same format as training covariate

    Returns: R-indices, Pattern c-indices between the selected repetition and all other repetitionss, Pattern c-indices among all repetitions, path to the final selected model used for deriving R-indices
    """
    if os.path.exists("%s/model_agreements.csv" % output_dir):
        agreement_f = pd.read_csv(os.path.join(output_dir, "model_agreements.csv"))
        if agreement_f["epoch"].max() < final_saving_epoch and (
            not (agreement_f["stop"] == "yes").any()
        ):
            raise Exception(
                "Waiting for other repetitions to finish to derive the final R-indices"
            )
        best_row = agreement_f.iloc[agreement_f["Rindices_corr"].idxmax()]
        if repetition > 3:
            max_index = best_row["best_model"]
            best_model_dir = os.path.join(output_dir, "model" + str(max_index))
            model = SurrealGAN()
            model.load(best_model_dir, best_row["epoch"])
            validation_data = parse_validation_data(  # type: ignore
                data,
                covariate,
                model.opt.correction_variables,  # type: ignore
                model.opt.normalization_variables,  # type: ignore
            )[1]
            r_indices = model.predict_rindices(validation_data)
        else:
            raise Exception(
                "At least 10 trained models are required (repetition number need to be at least 10)"
            )
    else:
        raise Exception(
            "Waiting for other repetitions to finish to derive the final R-indices"
        )
    return (
        np.array(r_indices),
        best_row["best_dimension_corr"],
        best_row["best_difference_corr"],
        best_row["dimension_corr"],
        best_row["difference_corr"],
        best_row["epoch"],
        best_model_dir,
    )


def repetitive_representation_learning(
    data: pandas.DataFrame,
    npattern: int,
    repetition: int,
    fraction: float,
    final_saving_epoch: int,
    output_dir: str,
    mono_loss_threshold: float = 0.006,
    saving_freq: int = 2000,
    recons_loss_threshold: float = 0.003,
    covariate: pandas.DataFrame = None,
    lam: float = 0.2,
    zeta: int = 80,
    kappa: int = 80,
    gamma: float = 2.0,
    mu: int = 500,
    eta: int = 6,
    alpha: float = 0.02,
    batchsize: int = 300,
    lipschitz_k: float = 0.5,
    verbose: bool = False,
    beta1: float = 0.5,
    lr: float = 0.0008,
    max_gnorm: int = 100,
    eval_freq: int = 100,
    start_repetition: int = 0,
    stop_repetition: int = 0,
    early_stop_thresh: float = 0.02,
) -> None:
    """
    Args:
    :param early_stop_thresh:
    :param alpha:
    :param covariate:
    :param saving_freq:
    :param data: dataframe, dataframe file with all ROI (input features) The dataframe contains
            the following headers: "
                                                             "i) the first column is the participant_id;"
                                                             "iii) the second column should be the diagnosis;"
                                                             "The following column should be the extracted features. e.g., the ROI features"
            covariate: dataframe, not required; dataframe file with all confounding covariates to be corrected. The dataframe contains
            the following headers: "
                                                             "i) the first column is the participant_id;"
                                                             "iii) the second column should be the diagnosis;"
                                                             "The following column should be all confounding covariates. e.g., age, sex"
    :param npattern: int, number of defined patterns
    :param repetition: int, number of repetition of training process
    :param fraction: float, fraction of data used for training in each repetition
    :param final_saving_epoch: int, epoch number from which the last model will be saved and model training will be stopped if saving criteria satisfied
    :param output_dir: str, the directory under which model and results will be saved
    :param mono_loss_threshold: float, chosen mono_loss threshold for stopping criteria
    :param recons_loss_threshold: float, chosen recons_loss threshold for stopping criteria
    :param lam: int, hyperparameter for orthogonal_loss
    :param zeta: int, hyperparameter for recons_loss
    :param kappa: int, hyperparameter for decompose_loss
    :param gamma: int, hyperparameter for change_loss
    :param mu: int, hyperparameter for mono_loss
    :param eta: int, hyperparameter for cn_loss
    :param batchsize: int, batch size for training procedure
    :param lipschitz_k: float, hyper parameter for weight clipping of transformation and reconstruction function
    :param verbose: bool, choose whether to print out training procedure
    :param beta1: float, parameter of ADAM optimization method
    :param lr: float, learning rate
    :param max_gnorm: float, maximum gradient norm for gradient clipping
    :param eval_freq: int, the frequency at which the model is evaluated during training procedure
    :param save_epoch_freq: int, the frequency at which the model is saved during training procedure
    :param start_repetition; int, indicate the last saved repetition index,
                                              used for restart previous half-finished repetition training or for parallel training; set defaultly to be 0 indicating a new repetition training process
    :param stop_repetition: int, indicate the index of repetition at which the training process early stop,
                                              used for stopping repetition training process early and resuming later or for parallel training; set defaultly to be None and repetition training will not stop till the end

    Returns: clustering outputs.

    """
    print("Start Surreal-GAN for semi-supervised representation learning")

    Surreal_GAN_model = Surreal_GAN_train(  # type: ignore
        npattern,
        final_saving_epoch,
        recons_loss_threshold,
        mono_loss_threshold,
        lam=lam,
        zeta=zeta,
        kappa=kappa,
        gamma=gamma,
        mu=mu,
        eta=eta,
        alpha=alpha,
        batchsize=batchsize,
        lipschitz_k=lipschitz_k,
        beta1=beta1,
        lr=lr,
        max_gnorm=max_gnorm,
        eval_freq=eval_freq,
        saving_freq=saving_freq,
        early_stop_thresh=early_stop_thresh,
    )

    if not stop_repetition:
        stop_repetition = repetition
    for i in range(start_repetition, stop_repetition):
        print("****** Starting training of Repetition " + str(i) + " ******")
        converge = Surreal_GAN_model.train(  # type: ignore
            data,
            covariate,
            output_dir,
            repetition,
            random_seed=i,
            data_fraction=fraction,
            verbose=verbose,
        )
        while not converge:
            print(
                "****** Model not converged at max interation, Start retraining ******"
            )
            converge = Surreal_GAN_model.train(  # type: ignore
                data=data,
                covariate=covariate,
                save_dir=output_dir,
                random_seed=i,
                data_fraction=fraction,
                verbose=verbose,
                repetition=repetition,
            )

    (
        r_indices,
        selected_model_dimension_corr,
        selected_model_difference_corr,
        dimension_corr,
        difference_corr,
        best_epoch,
        selected_model_dir,
    ) = representation_result(
        output_dir=output_dir,
        data=data,
        final_saving_epoch=final_saving_epoch,
        repetition=repetition,
        covariate=covariate,
    )

    pt_data = data.loc[data["diagnosis"] == 1][["participant_id", "diagnosis"]]

    for i in range(npattern):
        pt_data["r" + str(i + 1)] = r_indices[:, i]

    pt_data["Rindices-corr"] = ["%.3f" % ((dimension_corr + difference_corr) / 2)] + [
        "" for _ in range(r_indices.shape[0] - 1)
    ]
    pt_data["best epoch"] = [best_epoch] + ["" for _ in range(r_indices.shape[0] - 1)]
    pt_data["path to selected model"] = [selected_model_dir] + [
        "" for _ in range(r_indices.shape[0] - 1)
    ]
    pt_data["selected model Rindices-corr"] = [
        "%.3f" % ((selected_model_dimension_corr + selected_model_difference_corr) / 2)
    ] + ["" for _ in range(r_indices.shape[0] - 1)]
    pt_data["dimension-corr"] = ["%.3f" % (dimension_corr)] + [
        "" for _ in range(r_indices.shape[0] - 1)
    ]
    pt_data["difference-corr"] = ["%.3f" % (difference_corr)] + [
        "" for _ in range(r_indices.shape[0] - 1)
    ]
    pt_data["selected model dimension-corr"] = [
        "%.3f" % (selected_model_dimension_corr)
    ] + ["" for _ in range(r_indices.shape[0] - 1)]
    pt_data["selected model difference-corr"] = [
        "%.3f" % (selected_model_difference_corr)
    ] + ["" for _ in range(r_indices.shape[0] - 1)]

    pt_data.to_csv(os.path.join(output_dir, "representation_result.csv"), index=False)
    print("****** Surreal-GAN Representation Learning finished ******")
