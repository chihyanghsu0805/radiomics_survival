from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import pandas as pd

from run_pysurvival import run_pysurvival_with_repetitions, write_pysurvival_outputs

from datetime import datetime
from run_pyradiomics import get_pyradiomics
from sklearn.linear_model import LassoCV
from sklearn.utils import resample

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

def get_subject_scandate(features, scandate, subject):
  ''' Get Scan Dates for Pyradiomics'''
  scandate = features[scandate]
  scandate = scandate.values.tolist()
  date_list = []
  for i in scandate:
    datetime_object = datetime.strptime(i, '%m/%d/%Y').date()
    date_list.append(datetime_object.strftime('%Y')+datetime_object.strftime('%m')+datetime_object.strftime('%d'))

  subject = features[subject]
  subject = subject.values.tolist()
  subject_date = [x+'.'+y for x,y in zip(subject, date_list)]

  return subject_date

if __name__ == '__main__':
  ''' Clinical Features '''
  parser = argparse.ArgumentParser()
  parser.add_argument('--clinical_path', dest='clinical_path', type=str, required=True)
  parser.add_argument('--clinical_features', dest='clinical_features', type=str)
  parser.add_argument('--column_subject', dest='subject', type=str, required=True)
  parser.add_argument('--column_scandate', dest='scandate', type=str, required=True)
  parser.add_argument('--column_survival', dest='survival', type=str, required=True, default='OS')
  parser.add_argument('--column_event', dest='event', type=str, required=True, default='Dead')

  ''' Pyradiomics '''
  parser.add_argument('--radiomics_path', dest='radiomics_path', nargs='+')
  parser.add_argument('--radiomics_ext', dest='radiomics_ext', nargs='+')
  parser.add_argument('--radiomics_filters', dest='radiomics_filters', nargs='+', default='original')
  parser.add_argument('--radiomics_zscore', action='store_true', default=True)
  parser.add_argument('--radiomics_output', dest='radiomics_output')

  ''' Feature Selection for Radiomics '''
  parser.add_argument('--stable_path', dest='stable_path', type=str)
  parser.add_argument('--stable_p', dest='stable_p', type=float, default=0.001)
  parser.add_argument('--lasso', action='store_true', default=True)

  ''' Survival '''
  parser.add_argument('--survival_models', dest='models', nargs='+', type=str, required=True)
  parser.add_argument('--survival_repetitions', dest='repetitions', type=int, default=10)
  parser.add_argument('--survival_test_ratio', dest='test_ratio', type=float, default=0.1)
  parser.add_argument('--survival_output', dest='survival_output', type=str, required=True)
  args = parser.parse_args()

  ''' Read Clinical Features '''
  all_clinical_features_df = pd.read_csv(args.clinical_path)
  #if args.clinical_features) is not None:
  clinical_features = pd.read_csv(args.clinical_features)
  clinical_features = list(clinical_features.columns)
  print('Clinical Features:', clinical_features)

  clinical_features_df = all_clinical_features_df[clinical_features] # Categorical
  clinical_features_df = pd.get_dummies(clinical_features_df, columns=clinical_features)
  clinical_features_categorical = list(clinical_features_df.columns)
  all_clinical_features_df = pd.get_dummies(all_clinical_features_df, columns=clinical_features)

  subject_date = get_subject_scandate(all_clinical_features_df, args.scandate, args.subject)
  all_clinical_features_df = all_clinical_features_df.set_index(pd.Index(subject_date))

  ''' Radiomics '''
  num_channels = len(args.radiomics_path)
  print('Number of Radiomics Channels:', num_channels)

  if not os.path.isdir(args.radiomics_output):
    os.mkdir(args.radiomics_output)
  
  ''' Compute Pyradiomics '''
  # Radiomics are pre-computed in batches using command line.

  ''' Read Pyradiomics'''
  all_radiomics_dict = {}
  for i_subject in subject_date:
    subject_radiomics_dict = {}
    for radiomics_path, radiomics_ext in zip(args.radiomics_path, args.radiomics_ext):
      radiomics_file = radiomics_path+'/'+i_subject+radiomics_ext
      i_radiomics_df = pd.read_csv(radiomics_file)    
      i_radiomics_dict = i_radiomics_df.to_dict()
      i_radiomics_dict = filter_radiomics(i_radiomics_dict, args.radiomics_filters)
      i_radiomics_dict = add_prefix_radiomics(i_radiomics_dict, radiomics_ext)
      subject_radiomics_dict.update(i_radiomics_dict)
    subject_radiomics_dict = remove_duplicate_shape_features(subject_radiomics_dict)
    subject_radiomics_dict = remove_index(subject_radiomics_dict)
    all_radiomics_dict[i_subject] = subject_radiomics_dict

  all_radiomics_df = pd.DataFrame.from_dict(all_radiomics_dict, orient='index')
  all_radiomics_df.to_csv(args.radiomics_output+'/all_radiomics.csv')
  print('Number of Subjects:', len(all_radiomics_df.index))
  print('Number of Radiomic Features:', len(all_radiomics_df.columns))

  ''' Normalize Radiomics '''
  if args.radiomics_zscore:
    all_radiomics_df = normalize_features(all_radiomics_df)
    all_radiomics_df.to_csv(args.radiomics_output+'/normalized_radiomics.csv')
    print('Radiomic Features Zscored')

  ''' Stable Radiomics by Test-ReTest '''
  stable_features_df = pd.read_csv(args.stable_path)
  bool_significant = stable_features_df['p'] < args.stable_p
  stable_features = stable_features_df[bool_significant]['Radiomics']

  stable_radiomics_df = all_radiomics_df[stable_features]
  stable_radiomics_df.to_csv(args.radiomics_output+'/stable_radiomics.csv')
  print('Number of Stable Radiomic Features:', len(stable_features))

  ''' Radiomics Feature Selection with Lasso '''
  ''' Comment 
  radiomics_np = stable_radiomics_df.to_numpy()
  features = stable_radiomics_df.columns
  X = radiomics_np
  y = all_clinical_features[args.survival]
  bool_event = all_clinical_features[args.event] == 1
  X = X[bool_event]
  y = y[bool_event]

  predictive_features_counts = {}
  for i in range(100):
    X_bootstrap, y_bootstrap = resample(X,y,n_samples=50)
    #X_bootstrap = X
    #y_bootstrap = y
    clf = LassoCV(max_iter=1000, tol=0.01, cv=10).fit(X_bootstrap, y_bootstrap)
    if clf.score(X_bootstrap,y_bootstrap) > 0.9:
      #print(abs(clf.coef_))
      predictive_features = features[abs(clf.coef_) > 0]
      #print(predictive_features)
      for x in predictive_features:
        if x in predictive_features_counts:
         predictive_features_counts[x] = predictive_features_counts[x]+1
        else:
          predictive_features_counts[x] = 0

  predictive_features_counts_df = pd.DataFrame.from_dict(predictive_features_counts, orient='index')
  predictive_features_counts_df.to_csv(args.radiomics_output+'/predictive_radiomics.csv')
  print(predictive_features_counts_df)
  '''

  ''' Pysurvival '''
  df = pd.concat([stable_radiomics_df, all_clinical_features_df], axis=1)
  df = df.reset_index()
  outputs = run_pysurvival_with_repetitions(df, clinical_features_categorical, args.survival, args.event, args.models, args.test_ratio, args.repetitions)
  write_pysurvival_outputs(outputs, args.survival_output)
  print('Results written in:', args.survival_output)

