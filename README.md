# Surreal-GAN
Surreal-GAN is a semi-supervised representation learning method that is designed to identify disease-related heterogeneity among the patient group.  Surreal-GAN parses complex disease-related imaging patterns into low dimensional representation indices (r-indices), with each dimension indicating the severity of one relatively homogeneous imaging pattern.

The key point of the Surreal-GAN model is modeling disease as a continuous process and learning infinite transformation directions from CN to PT, with each direction capturing a specific combination of patterns and severity. The idea is realized by learning one transformation function, f, which takes both normal data and a continuous latent variable as inputs and outputs synthesized-PT data whose distribution is indistinguishable from that of real PT data. (As shown in the schematic diagram) Several different regularizations were introduced to further guide the transformation function. An inverse function g, jointly trained with f, is used for deriving R-indices after the training process.

![image info](./datasets/SURREAL-GAN.png)

## License
Copyright (c) 2016 University of Pennsylvania. All rights reserved. See[ https://www.cbica.upenn.edu/sbia/software/license.html](https://www.cbica.upenn.edu/sbia/software/license.html)

## Installation
We highly recommend that users install **Anaconda3** on their machines. After installing Anaconda3, Smile-GAN can be used following this procedure:

We recommend that users use the Conda virtual environment:


```bash
$ conda create --name surrealgan python=3.8
```
Activate the virtual environment

```bash
$ conda activate surrealgan
```

Install SurrealGAN from PyPi:

```bash
$ pip install SurrealGAN
```



## Input structure
The main function of SurrealGAN basically takes two Panda dataframes as data inputs, **data** and **covariate** (optional). Columns with the names *'participant_id'* and *'diagnosis'* must exist in both dataframes. Some conventions for the group label/diagnosis: -1 represents healthy control (CN) and 1 represents patient (PT); categorical variables, such as sex, should be encoded as numbers: Female for 0 and Male for 1, for example.

Example for **data**:

```bash
participant_id    diagnosis    ROI1    ROI2 ...
subject-1	    -1         325.4   603.4
subject-2            1         260.5   580.3
subject-3           -1         326.5   623.4
subject-4            1         301.7   590.5
subject-5            1	       293.1   595.1
subject-6            1         287.8   608.9
```
Example for **covariate**

```bash
participant_id    diagnosis    age    sex ...
subject-1	    -1         57.3   0
subject-2 	     1         43.5   1
subject-3           -1         53.8   1
subject-4            1         56.0   0
subject-5            1	       60.0   1
subject-6            1         62.5   0
```

## Example
We offer a toy dataset in the folder of SurrealGAN/dataset.

```bash
import pandas as pd
from SurrealGAN.Surreal_GAN_representation_learning import repetitive_representation_learning

train_data = pd.read_csv('train_roi.csv')
covariate = pd.read_csv('train_cov.csv')

output_dir = "PATH_OUTPUT_DIR"
npattern = 3
final_saving_epoch = 25000
max_epoch = 26000

## two important hyperparamters
lam = 0.2
gamma = 6
```

There are some hyper parameters need to be set by the user:

***batch\_size***: Size of the batch for each training epoch. (Default to be 100) It is **necessary** to be reset to 1/8 - 1/10 of the PT sample size.

***lam***: coefficient controlling the relative importance of cluster\_loss in the training objective function. (Default to be 0.2) It is **necessary** to try different ***lam*** values between 0.05 and 0.6 and use results that give the highest **pattern-agr-index** and **pattern-diff-agr-index** as returned in the output file.

***gamma***: coefficient controlling the relative importance of change\_loss in training objective function. (Default to be 6). It is **necessary** to try different ***gamma*** values between 0.5 and 8 and use results that give the highest **pattern-agr-index** and **pattern-diff-agr-index** as returned in the output file.

***saving\_freq***: the frequency (epochs) at which the model will be saved during the training process (Default to be 1000). One of the saved models will be determined to be optimal and used for deriving final results after the training procedure.


```bash				    
repetition_number = 20  # number of repetitions (at least 5 repetition\
	   is need to give the most reliable and reproducible result)
data_fraction = 1 # fraction of data used in each repetition
repetitive_representation_learning(train_data, npattern, repetition_number, data_fraction, final_saving_epoch, max_epoch, output_dir, \
		lr = 0.0008, batchsize=120, verbose=False, lipschitz_k=0.5, covariate= None, start_repetition=0, lam=lam, gamma = gamma)
```

**repetitive\_representation\_learning** is the **main** function for representation learning via Surreal-GAN. It performs the representation learning process repetitively with a pre-defined number of repetitions. Each repetition will save models in a subfolder called "repetition_i", in which the model will be saved every ***saving\_freq*** epochs before the final\_saving\_epoch while saving criteria are satisfied. The function will automatically choose the optimal saving epochs via **pattern-agr-index** and **pattern-diff-agr-index** (explained below) among the results.

After determining the optimal saving epoch, the repetition which has the highest agreement (measured by the sum of **pattern-agr-index** and **pattern-diff-agr-index**) with all other repetitions will be used to derive the final R-indices. Given the randomness in training procedure, it is necessary to run **at least 10 repetitions** to derive a reliable and reproducible result. 

Since the repetition training process may take a long training time on a normal desktop computer, the function enables an early stop and later resumption. Users can set ***stop\_repetition*** to be an early stopping point and ***start\_repetition*** to be the starting repetition index. This will also enable the user to run several repetitions in parallel.

The function automatically saves a CSV file with clustering results and returns the same dataframe. The dataframe also includes the **path to the final selected model** (the model used for deriving final R-indices), the average **pattern-agr-index** and **pattern-diff-agr-index** across all repetitively derived results, as well as the **selected model pattern-agr-index** and the **selected model pattern-diff-agr-index**.

**Two evaluation metrics used for measuring agreements between results and selecting the optimal model:**

* **pattern-agr-index**: With M dimensional R-indices derived by two different models defined as r<sup>1</sup> and r<sup>2</sup>, **pattern-agr-index** is defined as the average of M concordance indices for all dimensions, C(r<sup>1</sup><sub>i</sub>, r<sup>2</sup><sub>i</sub>).

* **pattern-diff-agr-index**: With M dimensional R-indices derived by two different models defined as r<sup>1</sup> and r<sup>2</sup>, **pattern-diff-agr-index** is defined as the average of M(M-1)/2 concordance indices for differences among all dimensions, C(r<sup>1</sup><sub>i</sub>-r<sup>1</sup><sub>j</sub>, r<sup>2</sup><sub>i</sub>-r<sup>2</sup><sub>j</sub>).

```					    
model_dir = 'PATH_TO_SAVED_MODEL' #the path to the final selected model (the one returned by function "repetitive_representation_learning")
r_indices = apply_saved_model(model_dir, train_data, covariate=None)
```
**apply\_saved\_model** is a function used for deriving representation (R-indices) for **new patient data** using a previously saved model. Input data and covariate (optional) should be Panda dataframes with the same format shown before. Only PT data, for which the users want to derive R-indices, needs to be provided with diagnoses set to 1. PT data can be any sample inside or outside of the training set. ***The function returns R-indices of PT data following the order of PT in the provided dataframe.***


## Citation
If you use this package for research, please cite the following paper:


```bash
@inproceedings{yang2022surrealgan,
title={Surreal-{GAN}:Semi-Supervised Representation Learning via {GAN} for uncovering heterogeneous disease-related imaging patterns},
author={Zhijian Yang and Junhao Wen and Christos Davatzikos},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=nf3A0WZsXS5}
}
```


