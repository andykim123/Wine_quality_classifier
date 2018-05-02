#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 19:50:16 2018

@author: Nigel
"""
import pandas as pd
import numpy as np
import os.path
import sys
import random
import subprocess
import enum
from scipy.stats import t

pd.options.mode.chained_assignment = None  # default='warn'


alpha = 0.005   # very strict test

# average train/test times for all models
lr_train_t = [0.399202108,0.401701927,0.426573992,0.420482874,0.336703062,0.387607098,0.346524954,0.373478889,0.35581398,0.379189968]
lr_test_t = [0.463749886,0.464840889,0.487515926,0.472535849,0.474876165,0.457687855,0.445669174,0.422504187,0.447098017,0.450365067]

dtree_train_t = [0.009735107,0.009175062,0.006947041,0.007826805,0.007400036,0.009325027,0.009768009,0.009634018,0.00824213,0.007286072]
dtree_test_t = [0.002659798,0.000579119,0.000582218,0.000622034,0.001368999,0.00065589,0.003829956,0.000641108,0.000732899,0.000620842]

adaboost_train_t = [0.06951499,0.064216852,0.070253134,0.078505039,0.073644161,0.070446968,0.068542957,0.079769135,0.086536169,0.072289944]
adaboost_test_t = [0.006875038,0.007534027,0.008028984,0.006933928,0.008401155,0.007076979,0.00722003,0.010015011,0.006700993,0.005442858]

rforest_train_t = [0.03392911,0.044325829,0.048328876,0.036793947,0.04406786,0.039087057,0.042352915,0.034124136,0.042324781,0.039099932]
rforest_test_t = [0.002074003,0.00270915,0.002177954,0.002640009,0.002212048,0.002313852,0.002172947,0.002403021,0.003190041,0.002204895]

kernelsvm_train_t =  [0.078768015,0.087855816,0.076040983,0.078831196,0.084443092,0.091758013,0.086833,0.073637962,0.080119133,0.087596893]
kernelsvm_test_t = [0.034939051,0.035103083,0.033616066,0.035727024,0.03508091,0.037729979,0.033043146,0.034065008,0.034037113,0.035366058]

gp_train_t = [3.614563942,2.467521906,1.991507053,2.016200066,2.136292934,2.094048023,1.98708415,2.052824974,1.921935081,1.952733994]
gp_test_t = [18.63989496,13.10904312,19.38328099,15.93073988,18.66968298,18.2545619,22.80483985,18.41597795,16.70865989,21.99465799]

nn_train_t = [9.8835,9.8667,10.5338,9.8234,10.9261,11.0287,9.7944,9.7396,10.1948,10.4661]
nn_test_t = [0.7328,0.7276,0.7332,0.7315,0.7188,0.94,0.7461,0.7219,0.7375,0.7224]

trainlist = [lr_train_t, dtree_train_t, adaboost_train_t, rforest_train_t, kernelsvm_train_t, gp_train_t, nn_train_t]
testlist = [lr_test_t, dtree_test_t, adaboost_test_t, rforest_test_t, kernelsvm_test_t, gp_test_t, nn_test_t]
class train(enum.Enum):
    lr_traintime = 0
    dtree_traintime = 1
    adaboost_traintime = 2
    rforest_traintime = 3
    kernelsvm_traintime = 4
    gp_traintime = 5
    nn_traintime = 6
class test(enum.Enum):
    lr_testtime = 0
    dtree_testtime = 1
    adaboost_testtime = 2
    rforest_testtime = 3
    kernelsvm_testtime = 4
    gp_testtime = 5
    nn_testtime = 6
    
# T test on training times
for i1 in range(0, len(trainlist)):         #model 1
    for i2 in range(i1+1, len(trainlist)):  #model 2 (excluding itself)
        t_score = (np.mean(trainlist[i1])-np.mean(trainlist[i2]))/np.sqrt((np.var(trainlist[i1])/len(trainlist[i1]))+(np.var(trainlist[i2])/len(trainlist[i2])))
        new_df = np.square((np.var(trainlist[i1])/len(trainlist[i1]))+(np.var(trainlist[i2])/len(trainlist[i2])))/((np.square(np.var(trainlist[i1]))/(np.square(len(trainlist[i1]))*(len(trainlist[i1])-1)))+(np.square((np.var(trainlist[i2])))/(np.square(len(trainlist[i2]))*(len(trainlist[i2])-1))))
        print("Comparing "+train(i1).name+" and "+train(i2).name+"\n")
        print("New degree of freedom: "+str(new_df)+"\n")
        print("Test T-Score: "+str(t_score)+"\n")
        if np.mean(trainlist[i1])<=np.mean(trainlist[i2]):
            compare_t = t.ppf(1-alpha,new_df)
            print("Comparable T-score: "+str(compare_t)+"\n")
            if t_score>=compare_t:
                print("Significantly, "+train(i1).name+" is better than "+train(i2).name+" / "+train(i1).name+" mean train time: "+str(np.mean(trainlist[i1]))+" / "+train(i2).name+" mean train time: "+str(np.mean(trainlist[i2]))+"\n")
            else:
                print("Statistically, no difference detected. But in this sample, "+train(i1).name+"is slightly better.\n"+train(i1).name+" mean train time: "+str(np.mean(trainlist[i1]))+"\n")
        else:
            compare_t = t.ppf(alpha,new_df)
            print("Comparable T-score: "+str(compare_t))
            if t_score>=compare_t:
                print("Significantly, "+train(i2).name+" is better than "+train(i1).name+" / "+train(i1).name+" mean train time: "+str(np.mean(trainlist[i1]))+" / "+train(i2).name+" mean train time: "+str(np.mean(trainlist[i2]))+"\n")
            else:
                print("Statistically, no difference detected. But in this sample, "+train(i2).name+"is slightly better.\n"+train(i2).name+" mean train time: "+str(np.mean(trainlist[i2]))+"\n")

# T test on testing times
for i1 in range(0, len(testlist)):         #model 1
    for i2 in range(i1+1, len(testlist)):  #model 2 (excluding itself)
        t_score = (np.mean(testlist[i1])-np.mean(testlist[i2]))/np.sqrt((np.var(testlist[i1])/len(testlist[i1]))+(np.var(testlist[i2])/len(testlist[i2])))
        new_df = np.square((np.var(testlist[i1])/len(testlist[i1]))+(np.var(testlist[i2])/len(testlist[i2])))/((np.square(np.var(testlist[i1]))/(np.square(len(testlist[i1]))*(len(testlist[i1])-1)))+(np.square((np.var(testlist[i2])))/(np.square(len(testlist[i2]))*(len(testlist[i2])-1))))
        print("Comparing "+test(i1).name+" and "+test(i2).name+"\n")
        print("New degree of freedom: "+str(new_df)+"\n")
        print("Test T-Score: "+str(t_score)+"\n")
        if np.mean(testlist[i1])<=np.mean(testlist[i2]):
            compare_t = t.ppf(1-alpha,new_df)
            print("Comparable T-score: "+str(compare_t)+"\n")
            if t_score>=compare_t:
                print("Significantly, "+test(i1).name+" is better than "+test(i2).name+" / "+test(i1).name+" mean test time: "+str(np.mean(testlist[i1]))+" / "+test(i2).name+" mean test time: "+str(np.mean(testlist[i2]))+"\n")
            else:
                print("Statistically, no difference detected. But in this sample, "+test(i1).name+" is slightly better.\n"+test(i1).name+" mean test time: "+str(np.mean(testlist[i1]))+"\n")
        else:
            compare_t = t.ppf(alpha,new_df)
            print("Comparable T-score: "+str(compare_t))
            if t_score>=compare_t:
                print("Significantly, "+test(i2).name+" is better than "+test(i1).name+" / "+test(i1).name+" mean test time: "+str(np.mean(testlist[i1]))+" / "+test(i2).name+" mean test time: "+str(np.mean(testlist[i2]))+"\n")
            else:
                print("Statistically, no difference detected. But in this sample, "+test(i2).name+" is slightly better.\n"+test(i2).name+" mean test time: "+str(np.mean(testlist[i2]))+"\n")

