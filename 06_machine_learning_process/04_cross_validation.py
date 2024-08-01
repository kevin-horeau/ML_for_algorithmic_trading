#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

# Import various cross-validation utilities from scikit-learn
from sklearn.model_selection import (train_test_split,
                                     KFold,
                                     LeaveOneOut,
                                     LeavePOut,
                                     ShuffleSplit,
                                     TimeSeriesSplit)

# Create a sample dataset of integers from 1 to 10
data = list(range(1, 11))
print(data)

# Perform a simple train-test split, with 80% of data for training
print(train_test_split(data, train_size=.8))

# Initialize K-Fold cross-validation with 5 splits
kf = KFold(n_splits=5)
# Iterate through the splits and print the indices for train and validate sets
for train, validate in kf.split(data):
    print(train, validate)

# Initialize K-Fold cross-validation with 5 splits, shuffling, and a random state for reproducibility
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Iterate through the splits and print the indices for train and validate sets
for train, validate in kf.split(data):
    print(train, validate)

# Initialize Leave-One-Out cross-validation
loo = LeaveOneOut()
# Iterate through the splits and print the indices for train and validate sets
for train, validate in loo.split(data):
    print(train, validate)

# Initialize Leave-P-Out cross-validation, leaving out 2 samples at a time
lpo = LeavePOut(p=2)
# Iterate through the splits and print the indices for train and validate sets
for train, validate in lpo.split(data):
    print(train, validate)

# Initialize ShuffleSplit cross-validation with 3 splits, test size of 2, and a random state
ss = ShuffleSplit(n_splits=3, test_size=2, random_state=0)
# Iterate through the splits and print the indices for train and validate sets
for train, validate in ss.split(data):
    print(train, validate)

# Initialize TimeSeriesSplit cross-validation with 5 splits
tscv = TimeSeriesSplit(n_splits=5)
# Iterate through the splits and print the indices for train and validate sets
for train, validate in tscv.split(data):
    print(train, validate)