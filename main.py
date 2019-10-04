import numpy as np
from utils import load_data
from utils import normalisation
from tslearn.barycenters import softdtw_barycenter
import matplotlib.pyplot as plt
from utils import get_labels
from utils import barrycenters
from utils import get_num_in_each_label
from utils import draw

np.random.seed(1)

xtrain,ytrain,xtest,ytest=load_data("GunPoint")
labels=get_labels(ytrain)
xtrain,xtest=normalisation(xtrain,xtest)
num_in_each_label=get_num_in_each_label(ytrain,labels)
barrycenters_array=barrycenters(xtrain,ytrain,labels,num_in_each_label)
draw(xtrain,ytrain,labels,barrycenters_array)