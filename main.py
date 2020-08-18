import numpy as np
from utils import load_data
from utils import normalisation
from tslearn.barycenters import softdtw_barycenter
import matplotlib.pyplot as plt
from utils import get_labels
from utils import barrycenters
from utils import get_num_in_each_label
from utils import draw, draw_animation
import os
import shutil

np.random.seed(1)

files=['Coffee']

for file_name in files:
	print(file_name)
	xtrain,ytrain,xtest,ytest=load_data(file_name)
	path='/media/hadi/laban/github_files/Soft_dtw_draw/results/'+file_name+'.png'
	labels=get_labels(ytrain)
	xtrain,xtest=normalisation(xtrain,xtest)
	num_in_each_label=get_num_in_each_label(ytrain,labels)
	barrycenters_array=barrycenters(xtrain,ytrain,labels,num_in_each_label)
	# draw(xtrain,ytrain,labels,barrycenters_array,path)
	draw_animation(xtrain,ytrain,labels,barrycenters_array)