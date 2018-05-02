import pandas as pd
import numpy as np
import os.path
import sys
import random
import subprocess
from scipy.stats import t
from sklearn.model_selection import ShuffleSplit

pd.options.mode.chained_assignment = None  # default='warn'

# function that checks the command line arguments
def validate_cmdline_args(nargs, msg):
    if len(sys.argv) < nargs:
        print(msg)
        sys.exit(1)

# function that checks whether the file names are valid with valid file paths
def validate_file_names(filename_1,filename_2,msg_1,msg_2):
	if not os.path.isfile(filename_1):
		print(msg_1)
		sys.exit(1)
	elif not os.path.isfile(filename_2):
		print(msg_2)
		sys.exit(1)

# function that checks whether the data file is valid with valid path
def validate_data_name(dataname,msg):
	if not os.path.isfile(dataname):
		print(msg)
		sys.exit(1)

# validate_cmdline_args(4,'Usage: python modelEvaluation.py <NAME OF MODEL_1 FILE> <NAME OF MODEL_2 FILE> <DATASET_PATH>')
# validate_file_names(sys.argv[1],sys.argv[2],"Invalid file name: "+sys.argv[1],"Invalid file name: "+sys.argv[2])
# validate_data_name(sys.argv[3],"Invalid data file name: "+sys.argv[3])
DATASET_PATH = sys.argv[1]

# model_1 = sys.argv[1]
# model_2 = sys.argv[2]

# list_1 = []
# list_2 = []

# Set alpha, a type-II error threshold.
# Usually, it is either 0.1, 0.05, or 0.01
alpha = 0.05

# list_2 = random.sample(xrange(100), 10)

data_features = ["fa","va","ca","rs","ch","fsd","tsd","dens","pH","sulp","alcohol","eval"]
data = pd.read_csv(DATASET_PATH,names=data_features)
#to_csv("/Users/dohoonkim/Desktop/cse517a/ApplicationProject/PCAdata.csv", sep=',',index=False)

var1 = 1
var2 = DATASET_PATH
var4 = var2.split("/")
var5 = len(var4)

print(len(data))

data_wow = data[1:160]

print(len(data_wow))

result=""

for i in range(1,var5-1):
    result = result +"/"+var4[i]


print(result)


