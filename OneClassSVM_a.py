# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 13:51:39 2017

@author: anurag
"""
#import dataset
import quandl
#ql = quandl.get("NASDAQOMX/OMXS30SHORTX2")

#import rest library
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt

#prepare classifier
classifier = OneClassSVM(kernel = 'linear')
pca = PCA(n_components = 5)

#fitting all the value
ql['Total Market Value'] = ql['Total Market Value']/max(ql['Total Market Value'])
ql['Dividend Market Value'] = ql['Dividend Market Value']/max(ql['Dividend Market Value'])

#remove all the nan valuue with 0
ql = ql.fillna(0)

#dong component analysis
new_ql = pca.fit_transform(ql)

#pushing all the vale in the support vector
classifier.fit(new_ql)
final_ql = classifier.predict(new_ql)
coef_ = classifier.coef_


