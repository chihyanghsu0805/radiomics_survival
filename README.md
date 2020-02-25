# Radiomics Survival Analysis
This repository performs survival analysis using clinical and radiomics features.
It utilizes two commonly used packages for computing radiomic features and survival model, pyradiomics and pysurvival.

## Pyradiomics
https://pyradiomics.readthedocs.io/en/latest/

## Pysurvival
https://square.github.io/pysurvival/

## Survival Analysis
Survival analysis predicts when an event is likely to happen.
It is essentially a regression model but with censored events.
Censoring refers to the cases that the event has not occurred.
A regression model would be biased toward the cases that did experience the event.
Therefore, the input is features (X), time to event (T), and event (E).
Event is a boolean that indicates whether it occurred or not.

Performance of a survival model is usually characterized by C-index and/or Integrated Brier Score.

## Clinical Features
These are assumed to be categorical and converted to one-hot.

## Radiomics
Radiomics converts unstructured data (images) into structured data (case x features).
However, the stability/reproducibilty is not well studied or poor due to the varying hardware and scanning parameters.
The stability is established by Test-Retest experiments with imaging within two weeks.
And this repository filters out radiomic features outside given p-value range.

Additionally, features can be filtered with other feature selection methods.
We used LASSO with bootstrap and determined a cutoff for predictiveness.

## Outputs
The final output is a csv file with each row as a model and the columns as its model performance.
