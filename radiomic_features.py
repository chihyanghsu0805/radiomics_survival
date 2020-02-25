from __future__ import absolute_import
from __future__ import print_function

import os
import pandas as pd

def get_radiomic_features(radiomics_path, radiomics_output, radiomics_ext, radiomics_filters, radiomics_zscore, subject_date):
  num_channels = len(radiomics_path)
  print('Number of Radiomics Channels:', num_channels)

  ''' Compute Pyradiomics '''
  # Radiomics are pre-computed in batches using command line.

  ''' Read Pyradiomics'''
  radiomics_dict = read_pyradiomics(subject_date, radiomics_path, radiomics_ext, radiomics_filters)
  if not os.path.isdir(radiomics_output):
    os.mkdir(adiomics_output)

  radiomics_df = pd.DataFrame.from_dict(radiomics_dict, orient='index')
  radiomics_df.to_csv(radiomics_output+'/all_radiomics.csv')

  print('Number of Subjects:', len(radiomics_df.index))
  print('Number of Radiomic Features:', len(radiomics_df.columns))

  ''' Normalize Radiomics '''
  if radiomics_zscore:
    radiomics_df = normalize_features(radiomics_df)
    radiomics_df.to_csv(radiomics_output+'/normalized_radiomics.csv')
    print('Radiomic Features Zscored')

  return radiomics_df

def filter_radiomics_by_lasso(radiomics_df, lasso_path, lasso_threshold, radiomics_output):
  ''' Predictive Radiomics by LASSO '''
  predictive_features_df = pd.read_csv(lasso_path)
  bool_significant = predictive_features_df['Counts'] > lasso_threshold
  predictive_features = predictive_features_df[bool_significant]['Features']

  predictive_radiomics_df = radiomics_df[predictive_features]
  predictive_radiomics_df.to_csv(radiomics_output+'/predictive_radiomics.csv')
  print('Number of Predictive Radiomic Features:', len(predictive_features))

  return predictive_radiomics_df

def filter_radiomics_by_stability(radiomics_df, stable_path, stable_p, radiomics_output):
  ''' Stable Radiomics by Test-ReTest '''
  stable_features_df = pd.read_csv(stable_path)
  bool_significant = stable_features_df['p'] < stable_p
  stable_features = stable_features_df[bool_significant]['Radiomics']

  stable_radiomics_df = radiomics_df[stable_features]
  stable_radiomics_df.to_csv(radiomics_output+'/stable_radiomics.csv')
  print('Number of Stable Radiomic Features:', len(stable_features))

  return stable_radiomics_df

def read_pyradiomics(subject_date, radiomics_path, radiomics_ext, radiomics_filters):
  radiomics_dict = {}
  for i_subject in subject_date:
    subject_radiomics_dict = {}
    for path, ext in zip(radiomics_path, radiomics_ext):
      radiomics_file = path+'/'+i_subject+ext
      i_radiomics_df = pd.read_csv(radiomics_file)
      i_radiomics_dict = i_radiomics_df.to_dict()
      i_radiomics_dict = filter_radiomics(i_radiomics_dict, radiomics_filters)
      i_radiomics_dict = add_prefix_radiomics(i_radiomics_dict, ext)
      subject_radiomics_dict.update(i_radiomics_dict)
    subject_radiomics_dict = remove_duplicate_shape_features(subject_radiomics_dict)
    subject_radiomics_dict = remove_index(subject_radiomics_dict)
    radiomics_dict[i_subject] = subject_radiomics_dict

  return radiomics_dict

def normalize_features(radiomics_df):
  mu = radiomics_df.mean(axis=0)
  sd = radiomics_df.std(axis=0)
  radiomics_df = radiomics_df.subtract(mu)
  radiomics_df = radiomics_df.div(sd)
  mu = radiomics_df.mean(axis=0)
  sd = radiomics_df.std(axis=0)

  return radiomics_df

def remove_index(radiomics_dict):
  for key, value in radiomics_dict.items():
    radiomics_dict[key] = value[0]

  return radiomics_dict

def remove_duplicate_shape_features(radiomics_dict):
  new_dict = {}
  key_seen = []
  for key, value in radiomics_dict.items():
    if 'shape' in key:
      key = key[3:]
    if key in key_seen:
      pass
    else:
      new_dict[key] = value
      key_seen.append(key)

  return new_dict

def add_prefix_radiomics(radiomics_dict, ext):
  ext = ext.split('.')
  pre = ext[1]
  new_dict = {}
  for key, value in radiomics_dict.items():
    new_key = pre+'_'+key
    new_dict[new_key] = value
  return new_dict

def filter_radiomics(radiomics_dict, filters):
  new_dict = {}
  for key, value in radiomics_dict.items():
    bool_filter = [x in key for x in filters]
    if any(bool_filter) and 'diagnostics' not in key:
      new_dict[key] = value
  return new_dict


