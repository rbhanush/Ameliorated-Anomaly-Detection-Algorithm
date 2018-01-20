import numpy as np
from random import shuffle
import matplotlib.font_manager
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import utils
from sklearn import metrics
from sklearn import preprocessing

#reading csv file using pandas read_csv function (generates a Data_Frame)
my_data = pd.read_csv('kdd99-unsupervised-ad.csv', names=['duration','protocol_type','service','flag','src_bytes',
	'dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell',
	'su_attempted:','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login',
	'is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','class_label'])
print my_data.shape
#print my_data
my_data = my_data.sample(frac=1).reset_index(drop=True)
my_data = my_data[:100000]
#--------------------------------Data Preprocessing starts------------------------------------------------------------

#--------------------------------------

#my_data.class_label.value_counts().plot(kind='bar')
#using only the data with relevant features


#Converting to One class SVM (make classification binary instead of multi-class)
#we need to use class 1 (normal) and class -1 (attack)

my_data.loc[my_data['class_label'] == "n", "class_label"] = 1
my_data.loc[my_data['class_label'] != 1, "class_label"] = -1

target_class = my_data['class_label']
#------------------------------------------

only_one_class = my_data.loc[my_data['class_label'] == 1]

#extracting the binary_class value as the target for training and testing.
#we select only a single column from the dataframe

#determine fraction of records that represent outliers ('target_class' represents a series)
outliers = target_class[target_class == -1]

outlier_records = my_data.loc[my_data['class_label'] == -1]
print outlier_records.shape
print "Outliers shape:\t",outliers.count()
print "target class count\t",target_class.count()

print "Fraction of outliers in data:\t",float(outliers.count())/float(target_class.count())

#we drop the 'class_label' column in order to perform unsupervised training with unlabelled data
#later on we compare the predicted class for test data with the respective labels stored in 'target_class' series

my_data.drop(["class_label"],axis=1, inplace=True)
outlier_records.drop(["class_label"],axis=1, inplace=True)
only_one_class.drop(["class_label"],axis=1, inplace=True)
#----------------------------------Data Preprocessing ends here---------------------------------------------------------------

#normalizing the data in order to reduce numerical instability
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(my_data)
my_data = pd.DataFrame(np_scaled)

#---#############---
np_scaled = min_max_scaler.fit_transform(only_one_class)
only_one_class = pd.DataFrame(np_scaled)
#---#############---


pca = PCA(n_components=10)
my_data = pca.fit_transform(my_data)
only_one_class = pca.fit_transform(only_one_class)

#----------------------------New modificaition on Wednesday 22 Nov 2017-------------------------------------------
out_recs = []
for i in range(len(my_data)):
	if target_class[i] == -1:
		out_recs.append(my_data[i])

#----------------------------New modificaition ENDS as on Wednesday 22 Nov 2017-------------------------------------------


print "pca done"
#---------------------------------------Generating model----------------------------------------------------------------------
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(my_data,target_class,train_size=0.6)#, random_state=48)
print x_train.shape

#nu --> represents an upperbound on the fraction of training errors and also a lower bound on the fraction of support vectors
#nu value ranges between 0 and 1
#it speaks about the fraction of outliers we expect in the data
#NOTE: many unsupervised learning algorithms require us to know the number of outliers or class members we expect

#kernel --> the kernel function to be used (to project the feature space to a higer dimension i.e. by using a non-linear kernel)
#most commonly used kernel function is the Gaussian Radial Basis Function (RBF)

#gamma --> parameter of RBF kernel, controls the influence of individual training samples (affecting the smoothness of model)
#low gamma --> imporves smoothness and generalizability of the model
#high gamma --> reduces smoothness and makes the model more tightly-fitted to the training data
#NOTE: experiment to find optimal value

#train the model

from sklearn import svm

nu = float(outliers.count()) / float(target_class.count())
acc_train=[]
prec_train=[]
recal_train=[]
f1_train=[]
auc_train=[]

acc_test=[]
prec_test=[]
recal_test=[]
f1_test=[]
auc_test=[]


x_train = np.asarray(x_train, dtype=int)
x_test = np.asarray(x_test, dtype=int)
y_train = np.asarray(y_train, dtype=int)
y_test = np.asarray(y_test, dtype=int)

nu = 0.2
print "nu:\t",nu
print len(only_one_class)
gama = [0.00009,0.00005,0.000069,0.000098,0.000111,0.000137]
for j in range(len(gama)):
	model = svm.OneClassSVM(nu=0.11, kernel='rbf', gamma=gama[j])
	print("uptil here")
	model.fit(only_one_class)
	#prediction on training data
	prediction_train = model.predict(x_train)
	target_train = y_train
	num_err_in_train_pred = prediction_train[prediction_train == -1].size

	#display results (for train set)
	print "----------working on training set(nu=",0.11,",gamma=",gama[j],")----------"
	print "pred ",type(prediction_train),"\ntarg ",type(target_train)
	acc_train.append(metrics.accuracy_score(target_train,prediction_train))
	prec_train.append(metrics.precision_score(target_train,prediction_train))
	recal_train.append(metrics.recall_score(target_train,prediction_train))
	f1_train.append(metrics.f1_score(target_train,prediction_train))
	auc_train.append(metrics.roc_auc_score(target_train,prediction_train))
	print "Accuracy:\t", acc_train[-1]
	print "Precision:\t", prec_train[-1]
	print "Recall:\t", recal_train[-1]
	print "F1:\t", f1_train[-1]
	print "AUC:\t", auc_train[-1]

	#prediciton on testing data
	prediction_test = model.predict(x_test)
	target_test = y_test
	num_err_in_test_pred = prediction_test[prediction_test == -1].size

	#display results (for test set)
	print "----------working on testing set(nu=",0.11,",gamma=",gama[j],")----------"
	acc_test.append(metrics.accuracy_score(target_test,prediction_test))
	prec_test.append(metrics.precision_score(target_test,prediction_test))
	recal_test.append(metrics.recall_score(target_test,prediction_test))
	f1_test.append(metrics.f1_score(target_test,prediction_test))
	auc_test.append(metrics.roc_auc_score(target_test,prediction_test))
	print "Accuracy:\t", acc_test[-1]
	print "Precision:\t", prec_test[-1]
	print "Recall:\t", recal_test[-1]
	print "F1:\t", f1_test[-1]
	print "AUC:\t", auc_test[-1]


	#predicting class of outliers
	print len(out_recs)
	prediction_outliers = model.predict(out_recs)
	#print prediction_outliers
	num_err_in_outlier_pred = prediction_outliers[prediction_outliers == 1].size
	
	print "Outlier accuracy\t", (1-(num_err_in_outlier_pred/prediction_outliers.size))


plt.figure(1)
plt.subplot(211)
plt.title("KDD Datatset - Train set")
plt.xlabel("Gamma")
plt.ylabel("Accuracy")
plt.plot(gama, f1_train)

plt.figure(2)
plt.subplot(211)
plt.title("KDD Dataset - Test set")
plt.xlabel("Gamma")
plt.ylabel("Accuracy")
plt.plot(gama, f1_test)

plt.show()
