import itertools
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

from .data_loading import (
    CNIterator,
    PTIterator,
    val_CN_construction,
    val_PT_construction,
)
from .model import SurrealGAN

__author__ = "Zhijian Yang"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Zhijian Yang"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Zhijian Yang"
__email__ = "zhijianyang@outlook.com"
__status__ = "Development"


def Covariate_correction(
    cn_data: np.ndarray, cn_cov: np.ndarray, pt_data: np.ndarray, pt_cov: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Eliminate the confound of covariate, such as age and sex, from the disease-based changes.
    :param cn_data: array, control data
    :param cn_cov: array, control covariates
    :param pt_data: array, patient data
    :param pt_cov: array, patient covariates
    :return: corrected control data & corrected patient data
    """
    min_cov = np.amin(cn_cov, axis=0)
    max_cov = np.amax(cn_cov, axis=0)
    pt_cov = (pt_cov - min_cov) / (max_cov - min_cov)
    cn_cov = (cn_cov - min_cov) / (max_cov - min_cov)
    beta = np.transpose(LinearRegression().fit(cn_cov, cn_data).coef_)
    corrected_cn_data = cn_data - np.dot(cn_cov, beta)
    corrected_pt_data = pt_data - np.dot(pt_cov, beta)
    correction_variables = {"max_cov": max_cov, "min_cov": min_cov, "beta": beta}
    return corrected_cn_data, corrected_pt_data, correction_variables


def Data_normalization(
    cn_data: np.ndarray, pt_data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Normalize all data with respect to control data to ensure a mean of 1 and std of 0.1
    among CN participants for each ROI
    :param cn_data: array, control data
    :param pt_data: array, patient data
    :return: normalized control data & normalized patient data
    """
    cn_mean = np.mean(cn_data, axis=0)
    cn_std = np.std(cn_data, axis=0)
    normalized_cn_data = 1 + (cn_data - cn_mean) / (10 * cn_std)
    normalized_pt_data = 1 + (pt_data - cn_mean) / (10 * cn_std)
    normalization_variables = {"cn_mean": cn_mean, "cn_std": cn_std}
    return normalized_cn_data, normalized_pt_data, normalization_variables


def apply_covariate_correction(
    data: np.ndarray, covariate: np.ndarray, correction_variables: dict
) -> np.ndarray:
    covariate = (covariate - correction_variables["min_cov"]) / (
        correction_variables["max_cov"] - correction_variables["min_cov"]
    )
    corrected_data = data - np.dot(covariate, correction_variables["beta"])
    return corrected_data


def apply_data_normalization(
    data: np.ndarray, normalization_variables: dict
) -> np.ndarray:
    normalized_data = 1 + (data - normalization_variables["cn_mean"]) / (
        10 * normalization_variables["cn_std"]
    )
    return normalized_data


def parse_train_data(
    data: np.ndarray,
    covariate: np.ndarray,
    random_seed: float,
    data_fraction: int,
    batch_size: int,
) -> Tuple:
    cn_data = (
        data.loc[data["diagnosis"] == -1]
        .drop(["participant_id", "diagnosis"], axis=1)
        .values
    )
    pt_data = (
        data.loc[data["diagnosis"] == 1]
        .drop(["participant_id", "diagnosis"], axis=1)
        .values
    )
    correction_variables = None
    if covariate is not None:
        cn_cov = (
            covariate.loc[covariate["diagnosis"] == -1]
            .drop(["participant_id", "diagnosis"], axis=1)
            .values
        )
        pt_cov = (
            covariate.loc[covariate["diagnosis"] == 1]
            .drop(["participant_id", "diagnosis"], axis=1)
            .values
        )
        cn_data, pt_data, correction_variables = Covariate_correction(
            cn_data, cn_cov, pt_data, pt_cov
        )
    normalized_cn_data, normalized_pt_data, normalization_variables = (
        Data_normalization(cn_data, pt_data)
    )
    cn_train_dataset = CNIterator(
        normalized_cn_data, random_seed, data_fraction, batch_size
    )
    pt_train_dataset = PTIterator(
        normalized_pt_data, random_seed, data_fraction, batch_size
    )
    return (
        cn_train_dataset,
        pt_train_dataset,
        correction_variables,
        normalization_variables,
    )


def parse_validation_data(
    data: np.ndarray,
    covariate: np.ndarray,
    correction_variables: dict,
    normalization_variables: dict,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    cn_data = (
        data.loc[data["diagnosis"] == -1]
        .drop(["participant_id", "diagnosis"], axis=1)
        .values
    )
    pt_data = (
        data.loc[data["diagnosis"] == 1]
        .drop(["participant_id", "diagnosis"], axis=1)
        .values
    )
    if cn_data.shape[0] != 0:
        if correction_variables is not None:
            cn_cov = (
                covariate.loc[covariate["diagnosis"] == -1]
                .drop(["participant_id", "diagnosis"], axis=1)
                .values
            )
            pt_cov = (
                covariate.loc[covariate["diagnosis"] == 1]
                .drop(["participant_id", "diagnosis"], axis=1)
                .values
            )
            cn_data = apply_covariate_correction(cn_data, cn_cov, correction_variables)
            pt_data = apply_covariate_correction(pt_data, pt_cov, correction_variables)
        normalized_cn_data = apply_data_normalization(cn_data, normalization_variables)
        normalized_pt_data = apply_data_normalization(pt_data, normalization_variables)
        cn_eval_dataset = val_CN_construction(normalized_cn_data).load()
        pt_eval_dataset = val_PT_construction(normalized_pt_data).load()
        return cn_eval_dataset, pt_eval_dataset
    else:
        if correction_variables is not None:
            pt_cov = (
                covariate.loc[covariate["diagnosis"] == 1]
                .drop(["participant_id", "diagnosis"], axis=1)
                .values
            )
            pt_data = apply_covariate_correction(pt_data, pt_cov, correction_variables)
        normalized_pt_data = apply_data_normalization(pt_data, normalization_variables)
        pt_eval_dataset = val_PT_construction(normalized_pt_data).load()
        return pt_eval_dataset


def apply_saved_model(
    model_dir: str,
    data: pandas.DataFrame,
    epoch: int,
    covariate: Optional[pandas.DataFrame],
) -> np.ndarray:
    """
    Function used for derive representation results from one saved model
    Args:
            model_dir: string, path to the saved data
            data: dataframe with same format as training data. PT data can be any samples in or out of the training set.
            covariate: data_frame, dataframe with same format as training covariate. PT data can be any samples in or out of the training set.
            epoch: int, the training epoch
    Returns: R-indices
    """
    data = data[data["diagnosis"] == 1]
    if covariate is not None:
        covariate = covariate[covariate["diagnosis"] == 1]
    model = SurrealGAN()
    model.load(model_dir, epoch)
    model.get_corr()  # type: ignore
    validation_data = parse_validation_data(
        data,
        covariate,
        model.opt.correction_variables,  # type: ignore
        model.opt.normalization_variables,  # type: ignore
    )
    model.predict_rindices(validation_data)
    return model.predict_rindices(validation_data)


def calculate_pair_wise_correlation(
    r1: np.ndarray, r2: np.ndarray, npattern: int
) -> Tuple[float, float]:
    # function for calculating pair-wise correlations between two saved models
    order_permutation = list(itertools.permutations(range(npattern)))
    corr = [0 for _ in range(npattern)]
    best_order = range(npattern)
    order_dic = {}
    for i in range(len(order_permutation)):
        order_correlation = [0 for _ in range(npattern)]
        for j in range(npattern):
            if str(j) + str(order_permutation[i][j]) not in order_dic:
                order_dic[str(j) + str(order_permutation[i][j])] = pearsonr(
                    r1[:, j], r2[:, order_permutation[i][j]]
                )[0]
            order_correlation[j] = order_dic[str(j) + str(order_permutation[i][j])]
        if np.mean(order_correlation) > np.mean(corr):
            corr = np.mean(order_correlation)
            best_order = order_permutation[i]  # type: ignore
    corr = [0 for _ in range(npattern)]
    for j in range(npattern):
        corr[j] = pearsonr(r1[:, j], r2[:, best_order[j]])[0]
    pairs = list(itertools.combinations(range(npattern), 2))
    diff_corr = 0
    for i in range(len(pairs)):
        diff_corr += pearsonr(
            r1[:, pairs[i][0]] - r1[:, pairs[i][1]],
            r2[:, best_order[pairs[i][0]]] - r2[:, best_order[pairs[i][1]]],
        )[0]
    return diff_corr / len(pairs), np.mean(corr)


def calculate_group_compare_correlation(
    prediction_rindices: List, npattern: int
) -> Tuple:
    # function for calculating dimension-correlation and difference-correlation among a groups of predicted r-indices
    diff_corr: List[List[float]] = [[] for _ in range(len(prediction_rindices))]
    dimension_corr: List[List[float]] = [[] for _ in range(len(prediction_rindices))]
    for i in range(len(prediction_rindices)):
        for j in range(i + 1, len(prediction_rindices)):
            if i != j:
                diff_c_indices, patt_c_indices = calculate_pair_wise_correlation(
                    prediction_rindices[i], prediction_rindices[j], npattern
                )
                diff_corr[i].append(diff_c_indices)
                diff_corr[j].append(diff_c_indices)
                dimension_corr[i].append(patt_c_indices)
                dimension_corr[j].append(patt_c_indices)
    diff_corr = [np.mean(diff_corr[_]) for _ in range(len(prediction_rindices))]
    dimnesion_corr = [
        np.mean(dimension_corr[_]) for _ in range(len(prediction_rindices))
    ]
    return diff_corr, dimnesion_corr


def check_multimodel_agreement(
    data: np.ndarray,
    covar: np.ndarray,
    output_dir: str,
    epoch: int,
    repetition: int,
    npattern: int,
) -> List:
    all_finish = True
    for i in range(repetition):
        path = output_dir + "/model" + str(i)
        if os.path.exists(path):
            for i in range(100000):
                try:
                    load_dic = torch.load(path)
                except Exception:
                    pass
                else:
                    break
            if epoch not in load_dic.keys():
                all_finish = False
                break
        else:
            all_finish = False
            break
    all_rindices = []
    if all_finish:
        for i in range(repetition):
            path = output_dir + "/model" + str(i)
            all_rindices.append(apply_saved_model(path, data, epoch, covariate=covar))
        pattern_diff_agr_index, pattern_agr_index = calculate_group_compare_correlation(
            all_rindices, npattern
        )
        return [pattern_diff_agr_index, pattern_agr_index]
    else:
        return []
