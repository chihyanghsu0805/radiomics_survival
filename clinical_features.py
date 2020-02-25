from __future__ import absolute_import
from __future__ import print_function

import pandas as pd

def get_clinical_features(clinical_data_df, clinical_features):

  ''' Read Clincal Feature List '''
  clinical_features_df = pd.read_csv(clinical_features)
  clinical_features_list = list(clinical_features_df.columns)
  print('Clinical Features:', clinical_features_list)  

  ''' Convert to Categorical '''
  clinical_features_df = clinical_data_df[clinical_features_list]
  clinical_features_df = pd.get_dummies(clinical_features_df, columns=clinical_features_list)
  #clinical_features_categorical = list(clinical_features_df.columns)
  return clinical_features_df



