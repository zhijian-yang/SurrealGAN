import pandas as pd
from Surreal_GAN.Surreal_GAN_representation_learning import repetitive_representation_learning
import os

if __name__ == '__main__':
	output_dir = './surrealgan_results'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	train_data = pd.read_csv('sample_data.csv')

	npattern = 3
	final_saving_epoch = 32000
	max_epoch = 33000


	repetitive_representation_learning(train_data, npattern, 10, 1, final_saving_epoch, max_epoch, output_dir, \
		lr = 0.0008, batchsize=120, verbose=True, lipschitz_k=0.5, covariate= None, start_repetition=0, lam=0.3)


