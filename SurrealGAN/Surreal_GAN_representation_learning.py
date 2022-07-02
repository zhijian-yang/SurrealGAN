import sys
import os
import numpy as np
import itertools
from sklearn import metrics
from .model import SurrealGAN
from .utils import parse_validation_data
from .training import Surreal_GAN_train
from lifelines.utils import concordance_index

__author__ = "Zhijian Yang"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Zhijian Yang"]
__license__ = "See LICENSE file"
__version__ = "0.0.1"
__maintainer__ = "Zhijian Yang"
__email__ = "zhijianyang@outlook.com"
__status__ = "Development"

def calculate_pair_wise_c_index(r1,r2,npattern):
	# function for calculating pair-wise c-index between two saved models
	order_permutation = list(itertools.permutations(range(npattern)))
	c_indices = [ 0 for _ in range(npattern)]
	best_order = range(npattern)
	for i in range(len(order_permutation)):
		order_c_indices = [ 0 for _ in range(npattern)]
		for j in range(npattern):
			order_c_indices[j] = concordance_index(r1[:,j],r2[:,order_permutation[i][j]])
		if np.mean(order_c_indices) > np.mean(c_indices):
			c_indices = order_c_indices
			best_order = order_permutation[i]
	pairs = list(itertools.combinations(range(npattern),2))
	pair_wise_c_indices = 0
	for i in range(len(pairs)):
		pair_wise_c_indices += concordance_index(r1[:,pairs[i][0]]-r1[:,pairs[i][1]],r2[:,best_order[pairs[i][0]]]-r2[:,best_order[pairs[i][1]]])
	return pair_wise_c_indices/len(pairs), np.mean(c_indices)


def calculate_group_compare_c_index(prediction_rindices,npattern):
	# function for calculating pattern c indices among a groups of predicted r-indices
	all_diff_c_indices = []
	all_pattern_c_indices = []
	for i in range(len(prediction_rindices)):
		local_pattern_index = []
		local_diff_index = []
		for j in range(len(prediction_rindices)):
			if i!=j:
				diff_c_indices, patt_c_indices = calculate_pair_wise_c_index(prediction_rindices[i],prediction_rindices[j],npattern)
				local_diff_index.append(diff_c_indices)
				local_pattern_index.append(patt_c_indices)
		all_diff_c_indices.append(np.mean(local_diff_index))
		all_pattern_c_indices.append(np.mean(local_pattern_index))
	return all_diff_c_indices, all_pattern_c_indices

def apply_saved_model(model_dir, data, covariate=None):
	"""
	Function used for derive representation results from one saved model
	Args:
		model_dir: string, path to the saved data
		data, data_frame, dataframe with same format as training data. PT data can be any samples in or out of the training set.
		covariate, data_frame, dataframe with same format as training covariate. PT data can be any samples in or out of the training set.
	Returns: R-indices

	"""
	data = data[data['diagnosis']==1]
	if covariate != None:
		covariate = covariate[covariate['diagnosis']==1]
	model = SurrealGAN()
	model.load(model_dir)
	validation_data = parse_validation_data(data, covariate,model.opt.correction_variables,model.opt.normalization_variables)
	model.predict_rindices(validation_data)
	return model.predict_rindices(validation_data)

def representation_result(model_dirs, npattern, data, final_saving_epoch, covariate=None):
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
	best_iteration_prediction_rindices = []
	best_pattern_c_index_mean = 0
	best_pattern_agr_index = []
	best_pattern_diff_agr_index = []
	best_epoch = 0

	if final_saving_epoch % 5000 == 0:
		save_epoch = [i * 5000 for i in range(2,final_saving_epoch//5000+1)]
	else:
		save_epoch = [i * 5000 for i in range(2,final_saving_epoch//5000+1)]+[final_saving_epoch]
	if len(model_dirs) > 9:
		for epoch in save_epoch:
			all_prediction_rindices = []
			for models in model_dirs:
				model = SurrealGAN()
				model.load(os.path.join(models, str(epoch)+'_epoch_model'))
				validation_data = parse_validation_data(data, covariate,model.opt.correction_variables,model.opt.normalization_variables)[1]
				all_prediction_rindices.append(model.predict_rindices(validation_data))
			pattern_diff_agr_index, pattern_agr_index = calculate_group_compare_c_index(all_prediction_rindices,npattern)
			if np.mean(pattern_agr_index)+np.mean(pattern_diff_agr_index) > best_pattern_c_index_mean:
				best_pattern_c_index_mean = np.mean(pattern_agr_index)+np.mean(pattern_diff_agr_index)
				best_pattern_agr_index = pattern_agr_index
				best_pattern_diff_agr_index = pattern_diff_agr_index
				best_iteration_prediction_rindices = all_prediction_rindices
				best_model_dir = os.path.join(model_dirs[pattern_agr_index.index(max(pattern_agr_index))], str(epoch)+'_epoch_model')
	
	else:
		raise Exception("At least 10 trained models are required (repetition number need to be at least 5)")
	max_index = best_pattern_agr_index.index(max(best_pattern_agr_index))
	return np.array(best_iteration_prediction_rindices[max_index]), max(best_pattern_agr_index), max(best_pattern_diff_agr_index), best_pattern_agr_index, best_pattern_diff_agr_index, best_model_dir


def repetitive_representation_learning(data, npattern, repetition, fraction, final_saving_epoch, max_epoch, output_dir, mono_loss_threshold=0.006,\
		recons_loss_threshold=0.003, covariate=None, lam=0.2, zeta=80, kappa=80, gamma=6, mu=500, eta=6, batchsize=100, lipschitz_k = 0.5, verbose = False, \
		beta1 = 0.5, lr = 0.0008, max_gnorm = 100, eval_freq = 50, save_epoch_freq = 5, start_repetition = 0, stop_repetition = None):
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
		max_epoch: int, maximum trainig epoch: training will stop even if criteria not satisfied.
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

	Surreal_GAN_model = Surreal_GAN_train(npattern, final_saving_epoch, max_epoch, recons_loss_threshold, mono_loss_threshold, \
		lam=lam, zeta=zeta, kappa=kappa, gamma=gamma, mu=mu, eta=eta, batchsize=batchsize, \
		lipschitz_k = lipschitz_k, beta1 = beta1, lr = lr, max_gnorm = max_gnorm, eval_freq = eval_freq, save_epoch_freq = save_epoch_freq)

	if stop_repetition == None:
		stop_repetition = repetition
	for i in range(start_repetition, stop_repetition):
		print('****** Starting training of Repetition '+str(i)+" ******")
		converge = Surreal_GAN_model.train(data, covariate, output_dir, random_seed=i, data_fraction = fraction, verbose = verbose)
		while not converge:
			print("****** Model not converged at max interation, Start retraining ******")
			converge = Surreal_GAN_model.train(data, covariate, output_dir, random_seed=i, data_fraction = fraction, verbose = verbose)

	saved_models = [os.path.join(os.path.join(output_dir,'repetition'+str(i)))  for i in range(repetition)]
	
	r_indices, selected_model_pattern_agr_index, selected_model_pattern_diff_agr_index, pattern_agr_cindex, pattern_diff_agr_cindex, selected_model_dir = representation_result(saved_models, npattern, data, final_saving_epoch, covariate = covariate)
	
	pt_data = data.loc[data['diagnosis'] == 1][['participant_id','diagnosis']]

	for i in range(npattern):
		pt_data['r'+str(i+1)] = r_indices[:,i]

	pt_data["path to selected model"] = [selected_model_dir]+['' for _ in range(r_indices.shape[0]-1)]
	pt_data["selected model pattern-agr-index"] = ["%.3f" %(selected_model_pattern_agr_index)]+['' for _ in range(r_indices.shape[0]-1)]
	pt_data["pattern-agr-index" ] = ["%.3f +- %.3f" %(np.mean(pattern_agr_cindex), np.std(pattern_agr_cindex))]+['' for _ in range(r_indices.shape[0]-1)]
	
	pt_data.to_csv(os.path.join(output_dir,'representation_result.csv'), index = False)
	print('****** Surreal-GAN Representation Learning finished ******')
