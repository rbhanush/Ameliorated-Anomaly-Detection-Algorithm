# - *- coding: utf-8 -*-
"""
Created on Tue Nov 14 00:28:44 2017

@author: gagla
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib
feat_labels = ['mean_radius','texture','perimeter','area','smoothness','compactness','concavity','concave_points',
	'symmetry','fractal_dimension','radius_se','14','15','16','17','18','19','20',
	'21','22','worst_radius','24','25','26','27','28','29','30',
	'actual_29']

my_data = pd.read_csv('breast-cancer-unsupervised-ad.csv', names=feat_labels)


print (my_data.shape)
my_data.loc[my_data['actual_29'] == "n", "actual_29"] = 1
my_data.loc[my_data['actual_29'] != 1, "actual_29"] = -1

target_class = my_data['actual_29']
print("Data is now uniform.")

outliers = target_class[target_class == -1]
outlier_records = my_data.loc[my_data['actual_29'] == -1]


#total notmal records
normal_records = my_data.loc[my_data['actual_29'] == 1]
normal = target_class[target_class == 1]
#print my_data.describe()


my_data.drop(["fractal_dimension","19","actual_29"],axis=1, inplace=True)
outlier_records.drop(["fractal_dimension","19","actual_29"],axis=1, inplace=True)
normal_records.drop(["fractal_dimension","19","actual_29"],axis=1, inplace=True)

print("Labels dropped")

#------Normalizing data------------------------------
x = normal_records.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
normal_records = pd.DataFrame(x_scaled)


x = outlier_records.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
outlier_records = pd.DataFrame(x_scaled)

x = my_data.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
my_data = pd.DataFrame(x_scaled)

#--------------------------------------------------
X = my_data.values
X = X.astype(float)

# Create y from output
y = target_class.values
y = y.astype(float)


orec = outlier_records.values
outlier_records = orec.astype(float)
orec1 = outliers.values
outliers = orec1.astype(float)

# Split the data into 40% test and 60% training
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,random_state=0)
print("Done splitting", X_train.shape)


from sklearn import svm

#nu = float(outliers.size) / float(target_class.size)
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

nu = 0.04

print ("nu:\t",nu)
gama = [0.00005,0.000037,0.00005,0.000069,0.000098,0.000111,0.000137]
for j in range(len(gama)):
	model = svm.OneClassSVM(nu=nu, kernel='rbf', gamma=gama[j])
	model.fit(X_train)
	#prediction on training data
	prediction_train = model.predict(X_train)
	target_train = y_train
	num_outlier_in_train_pred = prediction_train[prediction_train == -1].size

	#display results (for train set)
	print ("----------working on training set(nu=",nu,",gamma=",gama[j],")----------")
	acc_train.append(metrics.accuracy_score(target_train,prediction_train))
	prec_train.append(metrics.precision_score(target_train,prediction_train))
	recal_train.append(metrics.recall_score(target_train,prediction_train))
	f1_train.append(metrics.f1_score(target_train,prediction_train))
	auc_train.append(metrics.roc_auc_score(target_train,prediction_train))
	print ("Train Accuracy:\t", acc_train[-1])
	print ("Train Precision:\t", prec_train[-1])
	print ("Train Recall:\t", recal_train[-1])
	print ("Train F1:\t", f1_train[-1])
	print ("Train AUC:\t", auc_train[-1])

	#prediciton on testing data
	prediction_test = model.predict(X_test)
	target_test = y_test
	num_outlier_in_test_pred = prediction_test[prediction_test == -1].size

	#display results (for test set)
	print ("----------working on testing set(nu=",nu,",gamma=",gama[j],")----------")
    

	acc_test.append(metrics.accuracy_score(target_test,prediction_test))
	prec_test.append(metrics.precision_score(target_test,prediction_test))
	recal_test.append(metrics.recall_score(target_test,prediction_test))
	f1_test.append(metrics.f1_score(target_test,prediction_test))
	auc_test.append(metrics.roc_auc_score(target_test,prediction_test))
	print ("Test Accuracy:\t", acc_test[-1])
	print ("Test Precision:\t", prec_test[-1])
	print ("Test Recall:\t", recal_test[-1])
	print ("Test F1:\t", f1_test[-1])
	print ("Test AUC:\t", auc_test[-1])

	#predicting class of outliers
	prediction_outliers = model.predict(outlier_records)
	num_err_in_outlier_pred = prediction_outliers[prediction_outliers == -1].size
	outlier_acc = float(num_err_in_outlier_pred)/float(outliers.size)
    #display results (for outlier set)
	print ("----------working on training set(nu=",nu,",gamma=",gama[j],")----------")
	#print ("Train AUC:\t", auc_train[-1])
	print("Outlier acc",outlier_acc)


plt.figure(1)
plt.subplot(211)
plt.title("Breast cancer - Train set")
plt.xlabel("Gamma")
plt.ylabel("F1- score")
plt.plot(gama, f1_train)

plt.figure(2)
plt.subplot(211)
plt.title("Breast cancer - Test set")
plt.xlabel("Gamma")
plt.ylabel("F1 score")
plt.plot(gama, f1_test)

plt.show()
