from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import pandas as pd

from clinical_features import get_clinical_features
from radiomic_features import get_radiomic_features, filter_radiomics_by_stability, filter_radiomics_by_lasso
from run_pysurvival import run_pysurvival_with_repetitions, write_pysurvival_outputs

from datetime import datetime
from run_pyradiomics import get_pyradiomics
from sklearn.linear_model import LassoCV
from sklearn.utils import resample

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
  parser.add_argument('--lasso_path', dest='lasso_path', type=str)
  parser.add_argument('--lasso_threshold', dest='lasso_threshold', type=int)

  ''' Survival '''
  parser.add_argument('--survival_models', dest='models', nargs='+', type=str, required=True)
  parser.add_argument('--survival_repetitions', dest='repetitions', type=int, default=10)
  parser.add_argument('--survival_test_ratio', dest='test_ratio', type=float, default=0.1)
  parser.add_argument('--survival_output', dest='survival_output', type=str, required=True)
  args = parser.parse_args()

  ''' Initialize Data Frame with Survival and Event '''
  clinical_data_df = pd.read_csv(args.clinical_path)
  subject_date = get_subject_scandate(clinical_data_df, args.scandate, args.subject)
  clinical_data_df = clinical_data_df.set_index(pd.Index(subject_date))
  final_df = clinical_data_df[[args.survival, args.event]]
  #print(final_df)

  ''' Clinical Features '''
  if args.clinical_features is not None:
    # All Clinical Features were assumed to be categorical
    clinical_feature_df = get_clinical_features(clinical_data_df, args.clinical_features)
    final_df = pd.concat([final_df, clinical_feature_df], axis=1)

  ''' Radiomic Features '''
  if args.radiomics_path is not None:
    radiomics_df = get_radiomic_features(args.radiomics_path, \
                                         args.radiomics_output, \
                                         args.radiomics_ext, \
                                         args.radiomics_filters, \
                                         args.radiomics_zscore, \
                                         subject_date)
    if args.stable_path is not None:
      radiomics_df = filter_radiomics_by_stability(radiomics_df, args.stable_path, args.stable_p, args.radiomics_output)
    
    if args.lasso_path is not None:
      radiomics_df = filter_radiomics_by_lasso(radiomics_df, args.lasso_path, args.lasso_threshold, args.radiomics_output)    
    
    final_df = pd.concat([final_df, radiomics_df], axis=1)

  ''' Pysurvival '''
  final_df = final_df.reset_index()
  feature_list = list(final_df.columns)
  feature_list.remove(args.survival)
  feature_list.remove(args.event)
  feature_list.pop(0)
  print('Final Feature List:', feature_list)

  outputs = run_pysurvival_with_repetitions(final_df, feature_list, args.survival, args.event, args.models, args.test_ratio, args.repetitions)
  write_pysurvival_outputs(outputs, args.survival_output)
  print('Results written in:', args.survival_output)

