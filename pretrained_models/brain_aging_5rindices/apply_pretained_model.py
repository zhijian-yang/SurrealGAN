from SurrealGAN import Surreal_GAN_representation_learning
import pandas as pd
import numpy as np
from lifelines.utils import concordance_index
import itertools
from scipy.stats import pearsonr
import os

application_roi = pd.read_csv('ROI_volume_file.csv') # change to the path of CSV file with 72 ROI volumes
application_cov = pd.read_csv('covariate_file.csv') # change to the path of CSV file with sex and ICV volume as covarates
application_roi = application_roi[application_roi['diagnosis']==1].reset_index(drop=True)
application_cov = application_cov[application_cov['diagnosis']==1].reset_index(drop=True)

# apply the saved model to derive R-indices
model = os.path.join('saved_model') 
rindex = Surreal_GAN_representation_learning.apply_saved_model(model, application_roi, 50000, application_cov) 
representation_result = pd.DataFrame(data=rindex,columns=['r1','r2','r3','r4','r5'])
representation_result['participant_id'] = application_roi['participant_id']
representation_result.rename(columns={"r1": "r1", "r2": "r3",'r3':'r4','r4':'r5','r5':'r2'},inplace=True)
representation_result[['participant_id','r1','r2','r3','r4','r5']].to_csv('representation_result_all_istaging.csv',index=False) # change to the path/name of the output file


