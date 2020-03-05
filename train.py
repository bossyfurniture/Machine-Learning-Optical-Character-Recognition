"""
    Fundamentals of Machine Learning
    2019 Fall Project
    Group Name: ml
    Group Members: Abbas Furniturewalla, Jason Gibson, Marquita Ali, 
                   Omkar Mulekar

"""



'''========================================================================='''
''' Import Libraries'''
'''========================================================================='''

from sklearn.neural_network import MLPClassifier
import skimage.measure
from scipy import signal
import numpy as np
import pickle


'''========================================================================='''
''' Function Definitions'''
'''========================================================================='''

def BackToImg(list):
    '''
    Args:
        list of images that have been converted into a 1d array
    
    Returns:
        list of 2d arrays that can be viewed as an image
    '''
    im_size = 11
    images = np.zeros([im_size,im_size,len(list)])
    for i in range(len(list)):
        count = 0
        for r in range(im_size):
            for c in range(im_size):
                images[r,c,i] = list[i][count]
                count = count+1
        
    
    return images        

def ProcessData(data):
    '''
    Applies convolution, pooling and resizes images to a uniform length
    
    args:
        List of binary images
    
    Returns:
        List of images converted into 1D arrays
    '''
    blur = (1/9) * np.ones([3,3])
    Zdata = np.zeros([len(data)]).tolist()
    im_size = 11
    
    z = np.zeros([im_size,im_size])
    store = np.zeros([im_size,im_size,len(data)])
    
    for i in range(len(data)):
        a = signal.convolve2d(data[i],blur,boundary='symm',mode='full')
        a = skimage.measure.block_reduce(a, (5,5), func=np.max)
        data[i] = a
        R = np.shape(data[i])[0]
        C = np.shape(data[i])[1]
        for r in range(np.array([im_size,R]).min()):
            for c in range(np.array([im_size,C]).min()):
                store[r,c,i] = data[i][r][c] + z[r,c]
                
        Zdata[i] = store[:,:,i].flatten().tolist()
    
    x = Zdata, store
    return x

def load_pkl(fname):
    '''
    args:
        Path of pickle
        
    Return:
        list of elements in pickle
    '''
    
    with open(fname,'rb') as f:
        return pickle.load(f)

def predict_all(x_test,clf_all):
    s=clf_all.predict_proba(x_test)
    res = np.zeros(len(s))
    i = 0
    for row in s:
        index = max(max(np.argwhere(row == max(row))))
        if max(row)>0.5:
            res[i] = float(index)+1.
        else:
            res[i] = -1
        i = i+1
    return res

def save_pkl(fname,obj):
	with open(fname,'wb') as f:
		pickle.dump(obj,f)

'''========================================================================='''
''' Input and Process Data'''
'''========================================================================='''
def TrainModel(input_pkl=[],target_npy=[]):
    print("-----------------------------------------------------------------------")
    print("Fundamentals of Machine Learning")
    print("2019 Fall Project")
    print("Group Members: Abbas Furniturewalla, Jason Gibson,")
    print("               Marquita Ali, Omkar Mulekar")
    print("-----------------------------------------------------------------------")
    # Load in full data sets
    x_full_in = load_pkl(input_pkl)
    y_full = np.load(target_npy)
    print("Data Loaded!")
    
    y_full = y_full.tolist() # convert to python list
    
    num_each_in_full = 800 # number of each letter in full data set
    print("Processing Data...")
    
    # Perform preprocessing on data
    x_full_processed, x_full_processed_imgs = ProcessData(x_full_in)
    
    # Sort data
    x_full_sorted = [x for _,x in sorted(zip(y_full,x_full_processed))]
    y_full_sorted = [x for _,x in sorted(zip(y_full,y_full))]
    
    # Separate a's and b's from c's through k's
    x_ab_full = x_full_sorted[1:2*num_each_in_full]
    y_ab_full = y_full_sorted[1:2*num_each_in_full]
    
    '''========================================================================='''
    ''' Define and Train Model'''
    '''========================================================================='''
    
    # Neural network Architecture definition and training scheme specification
    clf_ab = MLPClassifier(activation='relu', 
                        alpha=1e-05, 
                        batch_size=200,
                        beta_1=0.9, # momentum parameter for adam
                        beta_2=0.999, # momentum parameter for adam
                        early_stopping=False,
                        epsilon=1e-08,
                        hidden_layer_sizes=(100,60,30,10), 
                        learning_rate='adaptive',
                        learning_rate_init=0.005,
                        max_iter=100000,
                        momentum=0.9, # not used for adam
                        n_iter_no_change=10,
                        nesterovs_momentum=True, # not used for adam
                        power_t=0.5,
                        random_state=1,
                        shuffle=True, # Shuffles samples
                        solver='adam',
                        tol=0.0001,
                        validation_fraction=0.1,
                        verbose=False,
                        warm_start=False)
    
    clf_all = MLPClassifier(activation='relu', 
                        alpha=1e-05, 
                        batch_size=200,
                        beta_1=0.9, # momentum parameter for adam
                        beta_2=0.999, # momentum parameter for adam
                        early_stopping=False,
                        epsilon=1e-08,
                        hidden_layer_sizes=(100,60,30,10), 
                        learning_rate='adaptive',
                        learning_rate_init=0.005,
                        max_iter=100000,
                        momentum=0.9, # not used for adam
                        n_iter_no_change=10,
                        nesterovs_momentum=True, # not used for adam
                        power_t=0.5,
                        random_state=1,
                        shuffle=True, # Shuffles samples
                        solver='adam',
                        tol=0.0001,
                        validation_fraction=0.1,
                        verbose=False,
                        warm_start=False)
    
    
    
    print("Training Network...")
    # Fit the network to the input data
    clf_ab.fit(x_ab_full, y_ab_full)
    clf_all.fit(x_full_processed,y_full)
    
    print("Saving networks to .sav files")
    pickle.dump(clf_ab,open('ab_model.sav','wb'))
    pickle.dump(clf_all,open('all_model.sav','wb'))
    
