# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 09:59:51 2019

@author: jbgib
"""
import pickle
from train import ProcessData, load_pkl, predict_all
import sys

'''

'''
def TestModel(filename, model):
    '''
    
    args:
        filename (String): Location of pickel containing binary images of 
        charecters
        model (String): 'ab' for testing the case of charecters a or b, 'all'
        for all charecters
        
    Returns:
        list of predictions 1., 2., 3., 4., 5., 6., 7., 8., corresponding 
        to a, b, c, d, h, i, j, k. 
    '''
    if(model == 'ab'):
        clf = pickle.load(open('ab_model.sav','rb'))
    elif(model == 'all'):
        clf = pickle.load(open('all_model.sav','rb'))
    else:
        sys.exit('ERROR: Please select model ab or all')
    try:
        data = load_pkl(filename)
    except OSError:
        print('cannot open', filename)
    
    data, data_imgs = ProcessData(data)
 
    if(model == 'ab'):
        pre = clf.predict(data)
    elif(model == 'all'):
        pre = predict_all(data,clf)
    else:
        pre = None
    
    return pre



