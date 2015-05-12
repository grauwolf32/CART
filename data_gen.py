import argparse
import numpy as np
import matplotlib.pyplot as pl

import collections
import  math
import copy
import itertools
import operator

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from numpy import linalg
import pylab

from sklearn import linear_model, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
from sklearn import cross_validation
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import metrics

def main():
    train_data = np.loadtxt('spam.train.txt')
    test_data  = np.loadtxt('spam.test.txt')

    tr_data = train_data[0::,1::]
    tr_answ = train_data[0::,0]

    ts_data = test_data[0::,1::]
    ts_answ = test_data[0::,0]

    data_file = open("train.data","w")
    answ_file = open("train.answ","w")

    m,n = tr_data.shape

    data_file.write(str(m)+" "+str(n)+"\n")

    np.savetxt("train.data",tr_data)
    np.savetxt("train.answ",tr_answ)

    data_file.close()
    answ_file.close()
  
    data_file = open("test.data","w")
    answ_file = open("test.answ","w")
  
    m,n = ts_data.shape
    data_file.write(str(m)+" "+str(n)+"\n")

    np.savetxt("test.data",ts_data)
    np.savetxt("test.answ",ts_answ)
    
    data_file.close()
    answ_file.close()
    	

if __name__ == "__main__":
    main() 

