import numpy as np
import scipy
import pandas as pd
import itertools
import torch 
from sklearn.preprocessing import minmax_scale
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn import metrics
from torch.autograd import Variable
from .data_loading import PTIterator, CNIterator, val_PT_construction, val_CN_construction
from lifelines.utils import concordance_index

__author__ = "Zhijian Yang"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Zhijian Yang"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Zhijian Yang"
__email__ = "zhijianyang@outlook.com"
__status__ = "Development"

def Covariate_correction(cn_data,cn_cov,pt_data,pt_cov):
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
    pt_cov = (pt_cov-min_cov)/(max_cov-min_cov)
    cn_cov = (cn_cov-min_cov)/(max_cov-min_cov)
    beta = np.transpose(LinearRegression().fit(cn_cov, cn_data).coef_)
    corrected_cn_data = cn_data-np.dot(cn_cov,beta)
    corrected_pt_data = pt_data-np.dot(pt_cov,beta)
    correction_variables = {'max_cov':max_cov,'min_cov':min_cov,'beta':beta}
    return corrected_cn_data, corrected_pt_data, correction_variables

def Data_normalization(cn_data,pt_data):
    """
    Normalize all data with respect to control data to ensure a mean of 1 and std of 0.1 
    among CN participants for each ROI
    :param cn_data: array, control data
    :param pt_data: array, patient data
    :return: normalized control data & normalized patient data
    """
    cn_mean = np.mean(cn_data,axis=0)
    cn_std = np.std(cn_data,axis=0)
    normalized_cn_data = 1+(cn_data-cn_mean)/(10*cn_std)
    normalized_pt_data = 1+(pt_data-cn_mean)/(10*cn_std)
    normalization_variables = {'cn_mean':cn_mean, 'cn_std':cn_std}
    return normalized_cn_data, normalized_pt_data, normalization_variables

def apply_covariate_correction(data, covariate, correction_variables):
    covariate = (covariate-correction_variables['min_cov'])/(correction_variables['max_cov']-correction_variables['min_cov'])
    corrected_data = data-np.dot(covariate,correction_variables['beta'])
    return corrected_data

def apply_data_normalization(data, normalization_variables):
    normalized_data = 1+(data-normalization_variables['cn_mean'])/(10*normalization_variables['cn_std'])
    return normalized_data

def parse_train_data(data, covariate, random_seed, data_fraction, batch_size):
    cn_data = data.loc[data['diagnosis'] == -1].drop(['participant_id','diagnosis'], axis=1).values
    pt_data = data.loc[data['diagnosis'] == 1].drop(['participant_id','diagnosis'], axis=1).values
    correction_variables = None
    if covariate is not None:
        cn_cov = covariate.loc[covariate['diagnosis'] == -1].drop(['participant_id', 'diagnosis'], axis=1).values
        pt_cov = covariate.loc[covariate['diagnosis'] == 1].drop(['participant_id','diagnosis'], axis=1).values
        cn_data,pt_data, correction_variables = Covariate_correction(cn_data,cn_cov,pt_data,pt_cov)
    normalized_cn_data, normalized_pt_data, normalization_variables = Data_normalization(cn_data,pt_data)
    cn_train_dataset = CNIterator(normalized_cn_data, random_seed, data_fraction, batch_size)
    pt_train_dataset = PTIterator(normalized_pt_data, random_seed, data_fraction, batch_size)
    return cn_train_dataset, pt_train_dataset, correction_variables, normalization_variables

def parse_validation_data(data, covariate, correction_variables, normalization_variables):
    cn_data = data.loc[data['diagnosis'] == -1].drop(['participant_id','diagnosis'], axis=1).values
    pt_data = data.loc[data['diagnosis'] == 1].drop(['participant_id','diagnosis'], axis=1).values
    if cn_data.shape[0] != 0:
        if correction_variables is not None:
            cn_cov = covariate.loc[covariate['diagnosis'] == -1].drop(['participant_id', 'diagnosis'], axis=1).values
            pt_cov = covariate.loc[covariate['diagnosis'] == 1].drop(['participant_id','diagnosis'], axis=1).values
            cn_data = apply_covariate_correction(cn_data, cn_cov, correction_variables)
            pt_data = apply_covariate_correction(pt_data, pt_cov, correction_variables)
        normalized_cn_data = apply_data_normalization(cn_data, normalization_variables)
        normalized_pt_data = apply_data_normalization(pt_data, normalization_variables)
        cn_eval_dataset = val_CN_construction(normalized_cn_data).load()
        pt_eval_dataset = val_PT_construction(normalized_pt_data).load()
        return cn_eval_dataset, pt_eval_dataset
    else:
        if correction_variables is not None:
            pt_cov = covariate.loc[covariate['diagnosis'] == 1].drop(['participant_id','diagnosis'], axis=1).values
            pt_data = apply_covariate_correction(pt_data, pt_cov, correction_variables)
        normalized_pt_data = apply_data_normalization(pt_data, normalization_variables)
        pt_eval_dataset = val_PT_construction(normalized_pt_data).load()
        return pt_eval_dataset



