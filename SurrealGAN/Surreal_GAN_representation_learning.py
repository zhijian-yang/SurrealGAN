import sys
import os
import numpy as np
import itertools
import pandas as pd
from sklearn import metrics
from .model import SurrealGAN
from .utils import parse_validation_data
from .training import Surreal_GAN_train
from scipy.stats import pearsonr

__author__ = "Zhijian Yang"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Zhijian Yang"]
__license__ = "See LICENSE file"
__version__ = "0.0.1"
__maintainer__ = "Zhijian Yang"
__email__ = "zhijianyang@outlook.com"
__status__ = "Development"

def apply_saved_model(model_dir, data, epoch, covariate=None):
	"""
	Function used for derive representation results from one saved model
	Args:
		model_dir: string, path to the saved data
		data, data_frame, dataframe with same format as training data. PT data can be any samples in or out of the training set.
		covariate, data_frame, dataframe with same format as training covariate. PT data can be any samples in or out of the training set.
	Returns: R-indices

	"""
	data = data[data['diagnosis']==1]
	if covariate is not None:
		covariate = covariate[covariate['diagnosis']==1]
	model = SurrealGAN()
	model.load(model_dir, epoch)
	model.get_corr()
	validation_data = parse_validation_data(data, covariate,model.opt.correction_variables,model.opt.normalization_variables)
	model.predict_rindices(validation_data)
	return model.predict_rindices(validation_data)

def representation_result(output_dir, npattern, data, final_saving_epoch, saving_freq, repetition, covariate=None):
	"""
	Function used for derive representation results from several saved models
	Args:
		model_dirs: list, list of dirs of all saved models
		npattern: int, number of pre-defined patterns
		data, data_frame, dataframe with same format as training data. 
		covariate, data_frame, dataframe with same format as training covariate. 
		final_saving_epoch: int, epoch number from which the last model will be saved and model training will be stopped if saving criteria satisfied
	Returns: R-indices, Pattern c-indices between the selected repetition and all other repetitionss, Pattern c-indices among all repetitions, path to the final selected model used for deriving R-indices

	"""
	if os.path.exists("%s/model_agreements.csv" % output_dir):
		agreement_f = pd.read_csv(os.path.join(output_dir,'model_agreements.csv'))
		if agreement_f['epoch'].max() < final_saving_epoch and (not (agreement_f['stop'] == 'yes').any()):
			raise Exception("Waiting for other repetitions to finish to derive the final R-indices")
		best_row = agreement_f.iloc[agreement_f['Rindices_corr'].idxmax()]
		if repetition > 3:
			max_index = best_row['best_model']
			best_model_dir = os.path.join(output_dir, 'model'+str(max_index))
			model = SurrealGAN()
			model.load(best_model_dir,best_row['epoch'])
			validation_data = parse_validation_data(data, covariate,model.opt.correction_variables,model.opt.normalization_variables)[1]
			r_indices = model.predict_rindices(validation_data)	
		else:
			raise Exception("At least 10 trained models are required (repetition number need to be at least 10)")
	else:
		raise Exception("Waiting for other repetitions to finish to derive the final R-indices")
	return np.array(r_indices), best_row['best_dimension_corr'], best_row['best_difference_corr'],  best_row['dimension_corr'], best_row['difference_corr'], best_row['epoch'], best_model_dir


def repetitive_representation_learning(data, npattern, repetition, fraction, final_saving_epoch, output_dir, mono_loss_threshold=0.006, saving_freq = 2000,\
		recons_loss_threshold=0.003, covariate=None, lam=0.2, zeta=80, kappa=80, gamma=2, mu=500, eta=6, alpha = 0.02, batchsize=300, lipschitz_k = 0.5, verbose = False, \
		beta1 = 0.5, lr = 0.0008, max_gnorm = 100, eval_freq = 100,  start_repetition = 0, stop_repetition = None, early_stop_thresh = 0.02):
	"""
	Args:
		data: dataframe, dataframe file with all ROI (input features) The dataframe contains
		the following headers: "
								 "i) the first column is the participant_id;"
								 "iii) the second column should be the diagnosis;"
								 "The following column should be the extracted features. e.g., the ROI features"
		covariate: dataframe, not required; dataframe file with all confounding covariates to be corrected. The dataframe contains
		the following headers: "
								 "i) the first column is the participant_id;"
								 "iii) the second column should be the diagnosis;"
								 "The following column should be all confounding covariates. e.g., age, sex"
		npattern: int, number of defined patterns
		repetition: int, number of repetition of training process
		fraction: float, fraction of data used for training in each repetition
		final_saving_epoch: int, epoch number from which the last model will be saved and model training will be stopped if saving criteria satisfied
		output_dir: str, the directory underwhich model and results will be saved
		mono_loss_threshold: float, chosen mono_loss theshold for stopping criteria
		recons_loss_threshold: float, chosen recons_loss theshold for stopping criteria
		lam: int, hyperparameter for orthogonal_loss
		zeta: int, hyperparameter for recons_loss
		kappa: int, hyperparameter for decompose_loss
		gamma: int, hyperparameter for change_loss
		mu: int, hyperparameter for mono_loss
		eta: int, hyperparameter for cn_loss
		batchsize: int, batck size for training procedure
		lipschitz_k: float, hyper parameter for weight clipping of transformation and reconstruction function
		verbose: bool, choose whether to print out training procedure
		beta1: float, parameter of ADAM optimization method
		lr: float, learning rate
		max_gnorm: float, maximum gradient norm for gradient clipping
		eval_freq: int, the frequency at which the model is evaluated during training procedure
		save_epoch_freq: int, the frequency at which the model is saved during training procedure
		start_repetition; int, indicate the last saved repetition index,
							  used for restart previous half-finished repetition training or for parallel training; set defaultly to be 0 indicating a new repetition training process
		stop_repetition: int, indicate the index of repetition at which the training process early stop,
							  used for stopping repetition training process eartly and resuming later or for parallel training; set defaultly to be None and repetition training will not stop till the end
		
	Returns: clustering outputs.

	"""
	print('Start Surreal-GAN for semi-supervised representation learning')

	Surreal_GAN_model = Surreal_GAN_train(npattern, final_saving_epoch, recons_loss_threshold, mono_loss_threshold, \
		lam=lam, zeta=zeta, kappa=kappa, gamma=gamma, mu=mu, eta=eta, alpha=alpha, batchsize=batchsize, \
		lipschitz_k = lipschitz_k, beta1 = beta1, lr = lr, max_gnorm = max_gnorm, eval_freq = eval_freq, saving_freq = saving_freq, early_stop_thresh = early_stop_thresh)

	if stop_repetition == None:
		stop_repetition = repetition
	for i in range(start_repetition, stop_repetition):
		print('****** Starting training of Repetition '+str(i)+" ******")
		converge = Surreal_GAN_model.train(data, covariate, output_dir, repetition, random_seed=i, data_fraction = fraction, verbose = verbose)
		while not converge:
			print("****** Model not converged at max interation, Start retraining ******")
			converge = Surreal_GAN_model.train(data, covariate, output_dir, random_seed=i, data_fraction = fraction, verbose = verbose)

	r_indices, selected_model_dimension_corr, selected_model_difference_corr, dimension_corr, difference_corr, best_epoch, selected_model_dir = representation_result(output_dir, npattern, data, final_saving_epoch, saving_freq, repetition, covariate = covariate)
	
	pt_data = data.loc[data['diagnosis'] == 1][['participant_id','diagnosis']]

	for i in range(npattern):
		pt_data['r'+str(i+1)] = r_indices[:,i]

	pt_data["Rindices-corr" ] = ["%.3f" %((dimension_corr+difference_corr)/2)]+['' for _ in range(r_indices.shape[0]-1)]
	pt_data["best epoch" ] = [best_epoch]+['' for _ in range(r_indices.shape[0]-1)]
	pt_data["path to selected model"] = [selected_model_dir]+['' for _ in range(r_indices.shape[0]-1)]
	pt_data["selected model Rindices-corr"] = ["%.3f" %((selected_model_dimension_corr+selected_model_difference_corr)/2)]+['' for _ in range(r_indices.shape[0]-1)]
	pt_data["dimension-corr" ] = ["%.3f" %(dimension_corr)]+['' for _ in range(r_indices.shape[0]-1)]
	pt_data["difference-corr" ] = ["%.3f" %(difference_corr)]+['' for _ in range(r_indices.shape[0]-1)]
	pt_data["selected model dimension-corr"] = ["%.3f" %(selected_model_dimension_corr)]+['' for _ in range(r_indices.shape[0]-1)]
	pt_data["selected model difference-corr"] = ["%.3f" %(selected_model_difference_corr)]+['' for _ in range(r_indices.shape[0]-1)]
	
	pt_data.to_csv(os.path.join(output_dir,'representation_result.csv'), index = False)
	print('****** Surreal-GAN Representation Learning finished ******')
