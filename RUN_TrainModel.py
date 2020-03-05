# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 14:42:34 2019

@author: jbgib
"""

# Import dependancy
from train import TrainModel

# Define TrainModel inputs
input = 'train_data.pkl'
target = "finalLabelsTrain.npy"

# Call TrainModel()
TrainModel(input_pkl=input, target_npy=target)