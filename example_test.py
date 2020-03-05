# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 14:42:34 2019

@author: jbgib
"""

# Import dependancy
from test import TestModel

# Examples of calling the function
ab = TestModel('train_data.pkl','ab')
al = TestModel('train_data.pkl','all')

# Print outputs
print(ab)
print(al)