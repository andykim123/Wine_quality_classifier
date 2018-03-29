import pandas as pd
import numpy as np
import os.path
import sys
import random
import subprocess
from scipy.stats import t

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

validate_cmdline_args(4,'Usage: python kernelSVM.py <NAME OF MODEL_1 FILE> <NAME OF MODEL_2 FILE> <DATASET_PATH>')
validate_file_names(sys.argv[1],sys.argv[2],"Invalid file name: "+sys.argv[1],"Invalid file name: "+sys.argv[2])
validate_data_name(sys.argv[3],"Invalid data file name: "+sys.argv[3])
DATASET_PATH = sys.argv[3]

model_1 = sys.argv[1]
model_2 = sys.argv[2]

list_1 = []
list_2 = []

# Set alpha, a type-II error threshold.
# Usually, it is either 0.1, 0.05, or 0.01
alpha = 0.005

# list_2 = random.sample(xrange(100), 10)

print("Files to be used:")
print("Model 1: "+model_1)
print("Model 2: "+model_2)
print("-----------------------")
print("Model Calculation Starts")
# execute 10 different cross-validations
for k in range(0,10):
    print("    "+str(k+1)+"th Run:")
    # proc = subprocess.Popen(['python', model_1,  DATASET_PATH, "true"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # print(proc.communicate()[0])
    # for each model, run a process and get the 10 cv results as a to_string format
    print("        "+model_1+":")
    proc1 = subprocess.check_output([sys.executable, model_1, DATASET_PATH, "true"])
    print("        "+model_2+":")
    proc2 = subprocess.check_output([sys.executable, model_2, DATASET_PATH, "true"])
    # parse the to_string format into 10 different string values
    temp_1 = proc1.split("[")[1].split("]")[0].split()
    temp_2 = proc2.split("[")[1].split("]")[0].split()
    for x in range(0,len(temp_1)):
    	# cast those string values into floats
	    list_1.append(float(temp_1[x]))
	    list_2.append(float(temp_2[x]))

print("Model Calculation Ends")
print("-----------------------")
# print(model_1+"- Mean: "+str(np.mean(list_1))+", Standard Deviation: "+str(np.sqrt(np.var(list_1))))
# conduct a two-sample two-tailed difference of mean t-test as we do not know the population variance
# also, we set equal_var as False as we cannot make any assumption that those two sets share the same population variance

# t_test_result = stats.ttest_ind(list_1,list_2,equal_var=False)
# print("P-Value: "+str(t_test_result.pvalue))
# # if p-value is smaller than alpha, we have statistical evidence that the means are different
# # then, we indicate which is larger
# if t_test_result.statistic<0 and t_test_result.pvalue<alpha:
#     print("Significantly, "+model_2+" is better than "+model_1+" / "+model_1+" mean accuracy: "+str(np.mean(list_1))+" / "+model_2+" mean accuracy: "+str(np.mean(list_2)))
# elif t_test_result.statistic>=0 and t_test_result.pvalue<alpha:
# 	print("Significantly, "+model_1+" is better than "+model_2+" / "+model_1+" mean accuracy: "+str(np.mean(list_1))+" / "+model_2+" mean accuracy: "+str(np.mean(list_2)))
# else:
# 	# if not, we do not have any statistical evidence that the means are different
# 	# however, we still indicate the user that which model have slightly better accuracy
# 	# but we make sure that that does not indicate the statistically significant difference
#     if t_test_result.static>=0:
#         print("Statistically, no difference detected. But in this sample, "+model_1+"is slightly better. "+model_1+" mean accuracy: "+str(np.mean(list_1)))
#     else:
#         print("Statistically, no difference detected. But in this sample, "+model_2+"is slightly better. "+model_2+" mean accuracy: "+str(np.mean(list_2)))

t_score = (np.mean(list_1)-np.mean(list_2))/np.sqrt((np.var(list_1)/len(list_1))+(np.var(list_2)/len(list_2)))
new_df = np.square((np.var(list_1)/len(list_1))+(np.var(list_2)/len(list_2)))/((np.square(np.var(list_1))/(np.square(len(list_1))*(len(list_1)-1)))+(np.square((np.var(list_2)))/(np.square(len(list_2))*(len(list_2)-1))))

print(list_1)
print(list_2)
print(np.mean(list_1))
print(np.mean(list_2))
print("New degree of freedom: "+str(new_df))
print("Test T-Score: "+str(t_score))

if np.mean(list_1)>=np.mean(list_2):
    compare_t = t.ppf(1-alpha,new_df)
    print("Comparable T-score: "+str(compare_t))
    if t_score>=compare_t:
        print("Significantly, "+model_1+" is better than "+model_2+" / "+model_1+" mean accuracy: "+str(np.mean(list_1))+" / "+model_2+" mean accuracy: "+str(np.mean(list_2)))
    else:
        print("Statistically, no difference detected. But in this sample, "+model_1+"is slightly better. "+model_1+" mean accuracy: "+str(np.mean(list_1)))
else:
	compare_t = t.ppf(alpha,new_df)
	print("Comparable T-score: "+str(compare_t))
	if t_score<=compare_t:
		print("Significantly, "+model_2+" is better than "+model_1+" / "+model_1+" mean accuracy: "+str(np.mean(list_1))+" / "+model_2+" mean accuracy: "+str(np.mean(list_2)))
	else:
		print("Statistically, no difference detected. But in this sample, "+model_2+"is slightly better. "+model_2+" mean accuracy: "+str(np.mean(list_2)))