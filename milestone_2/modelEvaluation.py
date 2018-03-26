import pandas as pd
import numpy as np
import os.path
import sys
import random
import subprocess
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from scipy import stats as stats

pd.options.mode.chained_assignment = None  # default='warn'

def validate_cmdline_args(nargs, msg):
    if len(sys.argv) < nargs:
        print(msg)
        sys.exit(1)

def validate_file_names(filename_1,filename_2,msg_1,msg_2):
	if not os.path.isfile(filename_1):
		print(msg_1)
		sys.exit(1)
	elif not os.path.isfile(filename_2):
		print(msg_2)
		sys.exit(1)

validate_cmdline_args(5,'Usage: python kernelSVM.py <NAME OF MODEL_1 FILE> <NAME OF MODEL_2 FILE> <DATASET_PATH>')
validate_file_names(sys.argv[1],sys.argv[2],"Invalid file name: "+sys.argv[1],"Invalid file name: "+sys.argv[2])
DATASET_PATH = sys.argv[3]

model_1 = sys.argv[1]
model_2 = sys.argv[2]

list_1 = []
list_2 = []

list_2 = random.sample(xrange(100), 10)

print("Files to be used:")
print("Model 1: "+model_1)
print("Model 2: "+model_2)
print("-----------------------")
print("Model Calculation Starts")
for k in range(0,10):
    print(str(k)+"th Run:")
    # proc = subprocess.Popen(['python', model_1,  DATASET_PATH, "true"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # print(proc.communicate()[0])
    proc1 = subprocess.check_output([sys.executable, model_1, DATASET_PATH, "true"])
    # proc2 = subprocess.check_output([sys.executable, model_2, DATASET_PATH, "true"])
    temp_1 = proc1.split("[")[1].split("]")[0].split()
    # temp_2 = proc2.split("[")[1].split("]")[0].split()
    for x in range(0,len(temp_1)):
	    list_1.append(float(temp_1[x]))
	    # list_2.append(float(temp_2[x]))

print("Model Calculation Ends")
# print(model_1+"- Mean: "+str(np.mean(list_1))+", Standard Deviation: "+str(np.sqrt(np.var(list_1))))
t_test_result = stats.ttest_ind(list_1,list_2,equal_var=False)
print(t_test_result.pvalue)
if t_test_result.statistic<0 and t_test_result.pvalue<0.05:
    print("Significantly, "+model_2+" is better than "+model_1)
elif t_test_result.statistic>=0 and t_test_result.pvalue<0.05:
	print("Significantly, "+model_1+" is better than "+model_2)
else:
    if t_test_result.static>=0:
        print("Statistically, no difference detected. But in this sample, "+model_1+"is slightly better")
    else:
        print("Statistically, no difference detected. But in this sample, "+model_2+"is slightly better")