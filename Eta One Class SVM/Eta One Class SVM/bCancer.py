import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# reading csv file using pandas read_csv function (generates a Data_Frame)
my_data = pd.read_csv ('breast-cancer-unsupervised-ad.csv',
                       names=['1', '2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31'])

print(my_data.shape)
# print my_data
# --------------------------ADI's code-------------------------------------
mydata = shuffle(my_data)

mydata.loc[mydata['31'] == "n", "31"] = 1
mydata.loc[mydata['31'] == "o", "31"] = -1
target_class = mydata['31']
mydata.drop (["31"], axis=1, inplace=True)

x_train,x_test,y_train,y_test = train_test_split(mydata,target_class,train_size=0.6)


#ab.to_csv('/Users/adityashah/Desktop/SML/As3/kdd99PCA.csv', encoding='utf-8', index=False)

x_train.to_csv('/Users/adityashah/Desktop/SML/As3/bCancer/x_train.csv', encoding='utf-8', index=False)
x_test.to_csv('/Users/adityashah/Desktop/SML/As3/bCancer/x_test.csv', encoding='utf-8', index=False)
y_train.to_csv('/Users/adityashah/Desktop/SML/As3/bCancer/y_train.csv', encoding='utf-8', index=False)
y_test.to_csv('/Users/adityashah/Desktop/SML/As3/bCancer/y_test.csv', encoding='utf-8', index=False)


