# Surreal-GAN
Surreal-GAN is a semi-supervised representation learning method that is designed to identify disease-related heterogeneity among the patient group. Surreal-GAN parses complex disease-related imaging patterns into low-dimensional representation indices (r-indices), with each dimension indicating the severity of one relatively homogeneous imaging pattern.

The key point of the Surreal-GAN model is modeling disease as a continuous process and learning infinite transformation directions from CN to PT, with each direction capturing a specific combination of patterns and severity. The idea is realized by learning one transformation function, f, which takes both normal data and a continuous latent variable as inputs and outputs synthesized-PT data whose distribution is indistinguishable from that of real PT data. As shown in the schematic diagram, several different regularizations were introduced to further guide the transformation function. An inverse function g, jointly trained with f, is used for deriving R-indices after the training process.

The fundamental framework of the basic Surreal-GAN model (Yang et al. 2022) inherently encourages independence among the derived R-indices, limiting its applicability in various scenarios. Therefore, starting from Surreal-GAN 0.1.1, we have implemented an updated Surreal-GAN model (Yang et al. 2023). This enhanced Surreal-GAN incorporates a correlation structure among the R-indices within a reduced representation latent space. This modification allows the model to capture interactions among multiple underlying neuropathological processes.
 
**We strongly encourage the users to upgrade to version 0.1.1.** 

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
The main function of SurrealGAN basically takes two Panda dataframes as data inputs: **data** and **covariate** (optional). Columns with the names *'participant_id'* and *'diagnosis'* must exist in both dataframes. Some conventions for the group label/diagnosis: -1 represents healthy control (CN) and 1 represents patient (PT); categorical variables, such as sex, should be encoded as numbers: Female for 0 and Male for 1, for example.

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

## Important Hyper-parameters

To ensure optimal performance and flexibility in Surreal-GAN representation learning, users can adjust the following hyper-parameters according to their specific needs:

### `batch_size`

- **Description:** Size of the batch for each training epoch
- **Default:** 300
- **Usage:** The default value is robust, but users can experiment and adjust based on Rindices-Correlation values.

### `lam`

- **Description:** Coefficient controlling the relative importance of `cluster_loss` in the training objective function.
- **Default:** 0.2
- **Usage:** Try different `lam` values between 0.05 and 1.6. Use the results yielding the highest Rindices-Correlation as indicated in the output file.

### `gamma`

- **Description:** Coefficient controlling the relative importance of `change_loss` in the training objective function.
- **Default:** 2
- **Usage:** Experiment with different `gamma` values between 0.1 and 8. Select the results with the highest Rindices-Correlation as returned in the output file.

### `saving_freq`

- **Description:** Frequency (in epochs) at which the model will be saved during the training process. At the end of the training process, one of the saved epochs will be determined to be optimal. The optimal epoch is returned in the output file and used for deriving final results after the training procedure.
- **Default:** 2000
- **Usage:** The users could select `saving_freq` based on the sample size and `final_saving_epoch`, (Recommend: 1/40-1/20 of `final_saving_epoch`). 

### `final_saving_epoch`

- **Description:** The last epoch during training at which the model is saved, and beyond which the training process stops.
- **Default:** NA
- **Usage:** Users are required to select the value for `final_saving_epoch` based on their specific datasets. While a larger value of `final_saving_epoch` can ensure reaching the optimal epoch, it comes at the expense of longer training times. As outlined in the section **Main function for Model Training**, the optimal epoch is determined at the conclusion of the training process. If users train all repetitions in parallel, real-time monitoring of agreements among models allows for early stopping, even before reaching the set `final_saving_epoch`. However, when training repetitions sequentially, it is advisable for users to carefully choose `final_saving_epoch` to ensure the attainment of the best model agreements (measured by **Rindices-Correlation**), (Recommend: 1500000*(300/patient\_sample\_size)).

## Rindices-Correlation

**Rindices-Correlation** is used as the metric for measuring agreements between results and selecting the optimal model. Specifically, it equals the means of the following two measurements:

* **Dimension-correlation**: With M-dimensional R-indices derived by two different models defined as r<sup>1</sup> and r<sup>2</sup>, **Dimension-correlation** is defined as the average of M Pearson’s correlations for all dimensions: 
$$\frac{1}{M}(\sum_{i=1}^M \rho(r_i^1,r_i^2))$$.

* **Difference-correlation**: With M-dimensional R-indices derived by two different models defined as r<sup>1</sup> and r<sup>2</sup>, **pattern-diff-agr-index** is defined as the average of M(M-1)/2 Pearson’s correlations for all pairs of dimensions: 
$$\frac{2}{M(M-1)}(\sum_{i=1}^M \sum_{j=i+1}^M \rho(r_i^1-r_j^1,r_i^2-r_j^2))$$

## Main function for Model Training
```bash				    
repetition_number = 30  # number of repetitions (at least 20 repetition\
	   is need to give the most reliable and reproducible result)
data_fraction = 1 # fraction of data used in each repetition
repetitive_representation_learning(train_data, npattern, repetition_number, data_fraction, final_saving_epoch, output_dir, \
		lr = 0.0008, batchsize=120, verbose=False, lipschitz_k=0.5, covariate= None, start_repetition=0, lam=lam, gamma = gamma)
```

The `repetitive_representation_learning` function is the cornerstone of representation learning using Surreal-GAN. It performs the repetitive training process with a user-defined number of repetitions.

### Process Description

- The function repetitively conducts the representation learning process with a pre-defined number of repetitions. Since representaiton learning is an unsupervised problem without ground truth, the agreements among repetitively trained models are used for evaluating model performance and selecting the optimal hyper-parameters. One of the repetitively trained models is determined to be optimal and used for deriving the final results. This process ensures the reliability and reproducibility of the derived R-indices. 

### Model Saving

- Repetitively trained models are saved in files named "model_i," with 'i' denoting the repetition index.
- Saving occurs every `saving_freq` epoch, preceding the `final_saving_epoch` while adhering to set criteria.

### Optimal Saving Epoch and Repetition

- The function automatically identifies the optimal saving epoch based on the highest mean **Rindices-Correlation** among the results.
- After determining the optimal epoch, the model with the highest agreement (measured by **Rindices-Correlation**) with all other repetitions will be used to derive the final R-indices.
- Given the randomness of the training procedure, it is necessary to run **at least 20 repetitions** to derive a reliable and reproducible result (i.e., set `repetition_numer` to be greater than 20). 

### Parallel vs. Sequential Training
Given the potentially prolonged duration of the repetitive training process on a standard desktop computer, the function provides an option for early stopping and later resumption. Users can set `stop_repetition` as an early stopping point and `start_repetition` to be the starting repetition index. 

- **Sequential Training**: When start_repetition is set to 1 and stop_repetition is set to the total repetition number (`repetition_number`), the function will train all repetitions sequentially. This may result in an extended training time.
- **Parellel Training**: Parallel Training: Setting start_repetition to 'i' and stop_repetition to 'i+1', where $1 \leq i \leq$ `repetition_number`, allows users to run multiple repetitions in parallel, particularly effective on HPC clusters.

### Monitoring Training Process
- Enabling verbose by setting it to True results in both printed updates of various losses and their saving in a 'results.txt' file. In most cases, verbose is set to False for efficiency.
- Agreements among repetitively trained models are calculated at every `saving_freq` epoch and are saved in a real-time updated CSV file named 'model_agreements.csv'.
 - In **Sequential Training**, the csv file will only be created and updated in the last repetition. 
 - In **Parellel Training**, the training processes of all repetitions interactively update the CSV file, allowing more efficient real-time monitoring.



## Output File
Upon completion of all repetitions, the function automatically saves a CSV file and returns the same dataframe. The CSV file contains the following information:

- **R-indices**: The derived R-indices of all PT participants in the training set. 
- **Rindices-corr**: Agreements among repetitively trained models with the current hyperparameters, `gamma` and `lam`.
- **best_epoch**: The most optimal epoch
- **path to selected model**: The path to the most optimal model.
- **selected model Rindices-corr**: Agreements between the optimal model and all the other repetitions.
- **Dimension-correlation** and **Difference-correlation**: Two metrics used for calculating **Rindices-corr**
- **selected model Dimension-correlation** and **selected model Difference-correlation**: Two metrics used for calculating **selected model Rindices-corr**

## Model Application to out-of-sample Participants
```					    
model_dir = 'PATH_TO_SAVED_MODEL' #the path to the final selected model (the one returned by function "repetitive_representation_learning")
r_indices = apply_saved_model(model_dir, application_data, epoch ,application_covariate=None)
```
**apply\_saved\_model** is a function used for deriving R-indices for **new patient data** using a previously saved model. 

### Input Data
- **Application data**: Only PT data, for which the users want to derive R-indices, needs to be provided with diagnoses set to 1. PT data can be any sample inside or outside of the training set. It should be in the form of Panda dataframes with the same format as training data. 
- **Application covariate**: (Optional) Users should provide the same covariate sets to the training data. It should be in the same format as the training covariates. 
- **model_dir**: The path to the model to be applied. Generally, it should be **path to selected model** provided in the output CSV file.
- **epoch**: The corresponding **best_epoch** provided in the output CSV file.

### Output
The function returns R-indices of PT data following the order of PT in the provided dataframe.


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


