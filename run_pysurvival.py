from __future__ import absolute_import
from __future__ import print_function

import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from pysurvival.models.semi_parametric import CoxPHModel
from pysurvival.models.semi_parametric import NonLinearCoxPHModel
from pysurvival.models.survival_forest import RandomSurvivalForestModel

from pysurvival.utils.metrics import concordance_index
from pysurvival.utils.metrics import integrated_brier_score 

def run_pysurvival_with_repetitions(data, features, survival, event, models, test_ratio, repetitions=10):

  num_samples = len(data.index)
  print('Number of Samples:', num_samples)

  ''' Initialize Outputs '''
  outputs = initialize_outputs(models, features)

  ''' Run Survival Model N times '''
  for _ in range(repetitions):
    
    ''' Dataset Splitting '''
    index_train, index_test = train_test_split(range(num_samples), test_size=test_ratio)
    data_train = data.loc[index_train].reset_index(drop=True)
    data_test = data.loc[index_test].reset_index(drop=True)

    X_train, X_test = data_train[features], data_test[features]
    T_train, T_test = data_train[survival].values, data_test[survival].values  
    E_train, E_test = data_train[event].values, data_test[event].values

    ''' Run Cox '''
    if 'cox' in models:
      coxph = CoxPHModel()
      coxph.fit(X_train, T_train, E_train, lr=0.0001, l2_reg=1e-2, init_method='zeros', verbose=False)
      c_index = concordance_index(coxph, X_test, T_test, E_test)
      outputs['cox']['c_index'].append(c_index)
      ibs = integrated_brier_score(coxph, X_test, T_test, E_test, t_max=None)
      outputs['cox']['ibs'].append(ibs)
      for idx, i in enumerate(features):
        outputs['cox']['weights'][i].append(coxph.weights[idx])

    ''' Run RSF '''
    if 'rsf' in models:
      rsf = RandomSurvivalForestModel(num_trees=200)
      rsf.fit(X_train, T_train, E_train, max_features="sqrt", max_depth=5, min_node_size=20)
      c_index = concordance_index(rsf, X_test, T_test, E_test)
      outputs['rsf']['c_index'].append(c_index)
      ibs = integrated_brier_score(rsf, X_test, T_test, E_test, t_max=None)
      outputs['rsf']['ibs'].append(ibs)
      for key, value in rsf.variable_importance.items():
        outputs['rsf']['importance'][key].append(value)

    ''' Run Deepsurv '''
    if 'deepsurv' in models:
      structure = [ {'activation': 'ReLU', 'num_units': 128}, {'activation': 'ReLU', 'num_units': 128}, {'activation': 'ReLU', 'num_units': 128}]

      nonlinear_coxph = NonLinearCoxPHModel(structure=structure)
      nonlinear_coxph.fit(X_train, T_train, E_train, lr=1e-4, init_method='xav_uniform', verbose = False)
      c_index = concordance_index(nonlinear_coxph, X_test, T_test, E_test)
      outputs['deepsurv']['c_index'].append(c_index)   
      ibs = integrated_brier_score(nonlinear_coxph, X_test, T_test, E_test, t_max=None)
      outputs['deepsurv']['ibs'].append(ibs)

  return outputs

def write_pysurvival_outputs(outputs_dict, outputs_path):
  all_results = []
  for model in outputs_dict.keys():
    metrics = outputs_dict[model].keys()
    for i in metrics:
      row_name = [model+'_'+i]
      results = outputs_dict[model][i]
      if isinstance(results, list):
        row = row_name+outputs_dict[model][i]
        all_results.append(row)
      if isinstance(results, dict):
        row = []
        for key, value in results.items():
          name = [x+'_'+key for x in row_name]
          row = name+value
          all_results.append(row)

  df = pd.DataFrame(all_results)
  df.to_csv(outputs_path, header=False, index=False)

def initialize_outputs(models, features):
  outputs = {}
  if 'cox' in models:
    outputs['cox'] = {}
    outputs['cox']['c_index'] = []
    outputs['cox']['ibs'] = []
    outputs['cox']['weights'] = {}
    for i in features:
      outputs['cox']['weights'][i] = []

  if 'rsf' in models:
    outputs['rsf'] = {}
    outputs['rsf']['c_index'] = []
    outputs['rsf']['ibs'] = []
    outputs['rsf']['importance'] = {}
    for i in features:
      outputs['rsf']['importance'][i] = []

  if 'deepsurv' in models:
    outputs['deepsurv'] = {}
    outputs['deepsurv']['c_index'] = []
    outputs['deepsurv']['ibs'] = []

  return outputs  

if __name__ == '__main__':

  ''' Parse Arguments '''
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', dest='data_path', type=str, required=True)
  parser.add_argument('--column_survival', dest='survival', type=str, required=True)
  parser.add_argument('--column_event', dest='event', type=str, required=True)
  parser.add_argument('--feature_list', dest='features', type=str)
  parser.add_argument('--models', dest='models', nargs='+', type=str, required=True)
  parser.add_argument('--repetitions', dest='repetitions', type=int, default=10)  
  parser.add_argument('--test_ratio', dest='test_ratio', type=int, default=0.1)
  parser.add_argument('--output_path', dest='output_path', type=str, required=True)
  args = parser.parse_args()

  ''' Read Data '''
  data = pd.read_csv(args.data_path)
  features = pd.read_csv(args.features)
  features = list(features.columns)
  print('Feature List:', features)  
  
  outputs = run_pysurvival_with_repetitions(data, features, args.survival, args.event, args.models, args.test_ratio, args.repetitions)
  write_pysurvival_outputs(outputs, args.output_path)
  print('Results written in:', args.output_path)

