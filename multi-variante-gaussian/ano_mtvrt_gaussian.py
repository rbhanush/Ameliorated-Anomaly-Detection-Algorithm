# README:
# 1. Python 2.6 and above (Python 2.7.5 using)
# 2. Using numpy, matplotlib, scipy, tensorflow, time, math, pandas and scipy third-party library
# 3. Just type "python XXX.py" XXX is the python file name.
# 4. The commented lines are the dubugging commands used for developing.



import numpy as np
from matplotlib import pyplot as plt
from scipy import io as spio
from tensorflow.python.framework import ops
import tensorflow as tf
import time
import scipy.io as scio
import math
import pandas
from matplotlib import pyplot as plt


# display_2d_data
def display_2d_data(X,marker):
    plt.plot(X[:,0],X[:,1],marker)
    plt.axis('square')
    return plt




ops.reset_default_graph()

# Create graph
sess = tf.Session()


# load data
# matlab data
loading = spio.loadmat('data1.mat')
# loading = spio.loadmat('musk.mat')
print("loading.keys: ",loading.keys())

data = loading['X']
x_val = loading['Xval']
y_val = loading['yval']   # y=1 means anormly
# y_val = loading['y']   # y=1 means anormly
# print("y_val: ", y_val)
print("data.shape: ", data.shape)
print("x_val.shape: ", x_val.shape)


# kdd data
# stock_dataframe = pandas.read_csv('kddcup.data_10_percent.csv')
# print(all_cols.size)
# col_num = all_cols.size
# num_rows = stock_dataframe.shape[0]

# harvard cancer data
# my_data = np.genfromtxt('pro_breast-cancer-unsupervised-ad.csv', delimiter=',')
# data = my_data[:, :-2]
# y_val = my_data[:, -1]
# print("data.shape: ", data.shape)



data_tf = tf.Variable(data)
#naive_gaussian(data_tf):
# mu = tf.reduce_mean(data_tf, axis=0)
# sigma = tf.keras.backend.var(data_tf, axis=0)

#parameters
num_features = data.shape[1]

# Placeholders
x_data_train = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
x_data_test = tf.placeholder(shape=[None, num_features], dtype=tf.float32)

x_phd = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
x_test_phd = tf.placeholder(shape=[None, num_features], dtype=tf.float32)

y_test_phd = tf.placeholder(shape=None, dtype=tf.float32)



# # min-max norm
# x_norm = tf.div(
#    tf.subtract(
#       x_phd, 
#       tf.reduce_min(x_phd)
#    ), 
#    tf.subtract(
#       tf.reduce_max(x_phd), 
#       tf.reduce_min(x_phd)
#    )
# )

# z-score norm
x_norm = tf.div(
   tf.subtract(
      x_phd, 
      tf.reduce_mean(x_phd)
   ), 
   (tf.keras.backend.var(x_phd, axis=0) + 0.0001)
)

# x_norm = x_phd

# # min-max norm
# x_test_norm = tf.div(
#    tf.subtract(
#       x_test_phd, 
#       tf.reduce_min(x_test_phd)
#    ), 
#    tf.subtract(
#       tf.reduce_max(x_test_phd), 
#       tf.reduce_min(x_test_phd)
#    )
# )

# z-score norm
x_test_norm = tf.div(
   tf.subtract(
      x_test_phd, 
      tf.reduce_mean(x_test_phd)
   ), 
   (tf.keras.backend.var(x_test_phd, axis=0) + 0.0001)
)


# x_test_norm = x_test_phd

# def estimate_gaussian(x_phd):
mu = tf.reduce_mean(x_norm, axis=0)
data_shifted = x_test_norm - mu
sigma = tf.keras.backend.var(x_norm, axis=0) + 0.0001


if (num_features>1):
    sigma_diag = tf.diag(sigma)

data_tf = tf.subtract(x_norm, tf.expand_dims(mu,1))


argu = tf.scalar_mul((2*np.pi)**(-num_features/2), tf.pow( tf.matrix_determinant(sigma_diag), (-0.5)))
sigma_inv = tf.matrix_inverse(sigma_diag)

power =  tf.multiply( tf.matmul(data_shifted, sigma_inv), (data_shifted))
# power =  tf.matmul(data_shifted, sigma_inv)

p_norm = (argu) *tf.exp(tf.scalar_mul(-0.5, tf.reduce_sum(power, 1)))
auc_tmp = tf.metrics.auc(y_test_phd, p_norm, num_thresholds=200)


init = tf.global_variables_initializer()
sess.run(init)
sess.run(tf.initialize_local_variables())


# 
print("mu: ", sess.run(mu, feed_dict={x_phd: data}))
print("sigma: ", sess.run(sigma, feed_dict={x_phd: data}))
# print(sess.run(mu,  feed_dict={x_phd: data}))
# print(sess.run(sigma, feed_dict={x_phd: data}))
print("sigma_diag :",sess.run((sigma_diag), feed_dict={x_phd: data}))
np.savetxt('sigma_diag.csv', sess.run((sigma_diag), feed_dict={x_phd: data}), delimiter = ',', fmt='%1.7f')

print("sigma_inv :",sess.run(sigma_inv, feed_dict={x_phd: data}))
# print("argu: ",sess.run(argu, feed_dict={x_phd: data}))
# print("data_shifted: ",sess.run(data_shifted, feed_dict={x_phd: data, x_test_phd: data}))
# print(sess.run(sigma_inv, feed_dict={x_phd: data}))
# print("power: ", sess.run(power, feed_dict={x_phd: data, x_test_phd: data}))
print("pnorm: ", sess.run(p_norm, feed_dict={x_phd: data, x_test_phd: data}))



def p_gaussian(train_np, test_np):
	p = sess.run(p_norm, feed_dict={x_phd: train_np, x_test_phd: test_np})
	return p



num_pionts = 1000
tp_np = np.zeros((1000))
fp_np = np.zeros((1000))
fn_np = np.zeros((1000))
tn_np = np.zeros((1000))
precision_np = np.zeros((1000))
recall_np = np.zeros((1000))
f1_np = np.zeros((1000))

def find_threshold(p, y_train):
	best_threshold = 0.
	best_f1 = 0.
	F1 = 0.
	step = (np.max(p)-np.min(p))/num_pionts
	total = p.shape[0]
	idx = 0
	for epsilon in np.arange(np.min(p),np.max(p),step):
		cvPrecision = p<epsilon
		print("cvPrecision: ", cvPrecision.shape)
		tp = np.sum((cvPrecision == 0) & (y_train.reshape([-1]) == 0)).astype(float)  
		print("tp: ", tp, total, y_train.shape)
		# print(((cvPrecision == 1) & (y_train == 1)).shape)
		fp = np.sum((cvPrecision == 0) & (y_train.reshape([-1]) == 1)).astype(float)  
		print("fp: ", fp, total, y_train.shape)
		fn = np.sum((cvPrecision == 1) & (y_train.reshape([-1]) == 0)).astype(float)  
		tn = np.sum((cvPrecision == 1) & (y_train.reshape([-1]) == 1)).astype(float)  
		precision = tp/(tp+fp)  
		recall = tp/(tp+fn)   
		F1 = (2*precision*recall)/(precision+recall)  # F1Score

		tp_np[idx] = tp
		fp_np[idx] = fp
		fn_np[idx] = fn
		tn_np[idx] = tn
		precision_np[idx] = precision
		recall_np[idx] = recall
		f1_np[idx] = F1

		if F1 > best_f1:  
			best_f1 = F1
			print(best_f1)
			best_threshold = epsilon
		idx += 1
	return best_threshold,best_f1



def ano_detect(threshold, p):
	outliers = np.where(p<threshold) 
	return outliers


# p = p_gaussian(data, x_val)
p = p_gaussian(data, data)

best_threshold, best_f1 = find_threshold(p, y_val)

print(best_threshold)
print('best_f1: ', best_f1)


# The outliers are exactly anormlous data in the dataset

outliers = ano_detect(best_threshold, p)
print(outliers)

# print("auc: ", sess.run(auc_tmp, feed_dict={x_phd: data, x_test_phd: data, y_test_phd: y_val}))
tpr_np = tp_np/(tp_np + fn_np)
fpr_np = fp_np/(fp_np + tn_np)


plt.plot(data[outliers,0],data[outliers,1],'o',markeredgecolor='r',markerfacecolor='w',markersize=10.)
plt = display_2d_data(data, 'bx')
plt.show()



