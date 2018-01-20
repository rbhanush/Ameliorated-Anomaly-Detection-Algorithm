import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import pandas
from itertools import product
from sklearn.neighbors import KNeighborsClassifier
from time import time
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import decomposition
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold
import operator
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def randomForest(train_data, train_label, num_features):
    model1 = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    model1.fit(train_data, train_label)

    # Extract highly relevant features
    feat_labels = np.asarray(num_features)
    feat_labels = feat_labels[:30]
    t = np.asarray(model1.feature_importances_.astype(float))
    new_feat = np.column_stack((feat_labels, t))
    y = new_feat[:, 1].astype(np.float)
    y = y.reshape(30, 1)
    new_feat = np.delete(new_feat, 1, 1)
    new_feat = np.hstack((new_feat, y))
    sorted_feat = sorted(new_feat, key=lambda x: str(x[1]))
    new_feat_descending = np.array(sorted_feat)[::1]
    final = []
    for i in range(len(new_feat_descending)):
        if not 'e-' in new_feat_descending[i, 1]:
            final.append(new_feat_descending[i, :])
    imp_feat = np.asarray(final[:10])
    imp_feat_df = pd.DataFrame(imp_feat, index=None)

    return imp_feat_df

def updateLabels(dataset):
    dataset.loc[dataset['label'] == "n", "label"] = 1
    dataset.loc[dataset['label'] == "o", "label"] = 0

    return dataset


def plotting1(k_range, k_accuracies):
    plt.plot(k_range, k_accuracies)
    plt.xlabel('Value of K')
    plt.ylabel('Cross Validated Averaged Accuracy')

    plt.show()

def plotting2(data, label, k):
    X = data[:, :2]
    y = label
    n_neighbors = k
    h = 0.02

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    model = neighbors.KNeighborsClassifier(n_neighbors, weights='distance', algorithm='ball_tree', leaf_size=100, metric='euclidean')
    model.fit(X, y)

    x1, x2 = X[:, 0].min() - 1, X[:, 0].max() + 1
    y1, y2 = X[:, 0].min() - 1, X[:, 0].max() + 1
    x3, y3 = np.meshgrid(np.arange(x1, x2, h), np.arange(y1, y2, h))
    Z = model.predict(np.c_[x3.ravel(), y3.ravel()])

    Z = Z.reshape(x3.shape)
    plt.figure()
    plt.pcolormesh(x3, y3, Z, cmap = cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='g', s=16)
    # cmap=cmap_bold
    plt.xlim(x3.min(), x3.max())
    plt.ylim(y3.min(), y3.max())
    plt.title("Classification when K = %i" % (n_neighbors))

    plt.show()

def main():
    col_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
                 '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', 'label']

    my_data = pandas.read_csv("breast-cancer-unsupervised-ad.csv", header=None, names=col_names)

    num_features = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
                    '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']

    my_data = updateLabels(my_data)

    train_data = my_data[num_features].astype(float)
    train_data1 = my_data[num_features].astype(float)

    labels1 = my_data['label'].copy()
    # print(labels1.value_counts())
    target_class1 = my_data['label']
    train_label = target_class1.values
    train_label = train_label.astype(float)

    df = pd.DataFrame()
    a = randomForest(train_data, train_label, num_features)
    for feature in a[0]:
        df = pd.concat([df, train_data[feature]], axis=1)
    print("Random Forest Completed")
    print(df.shape)

    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    print("Scaling for Train Set Completed")

    pca = decomposition.PCA(n_components=5)
    df = pca.fit_transform(df)
    print(df.shape)

    df1, df2, Y_train, Y_test = train_test_split(df, train_label, test_size=0.4)

    acc_train = []
    prec_train = []
    recal_train = []
    f1_train = []

    acc_test = []
    prec_test = []
    recal_test = []
    f1_test = []

    ###############################KFold and Cross Validation###########################################################

    maxK = 0
    k_range = range(1, 11)
    k_accuracies = []
    for k in k_range:
        print(k)
        knn = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree', leaf_size=100, metric='euclidean')
        print('kNN')
        scores = cross_val_score(knn, df1, Y_train, cv=5, scoring='accuracy')
        # print(scores)
        k_accuracies.append(scores.mean())
    print('\n')
    print(k_accuracies)
    print('\n')

    max_index, max_value = max(enumerate(k_accuracies), key=operator.itemgetter(1))
    maxK = max_index + 1
    max_value = max_value * 100
    print('Optimal Value of K is ' + repr(maxK))
    print('Maximum Cross Validated Averaged Accuracy when K = ' + repr(maxK) + ' is ' + repr(max_value))
    print('\n')
    print('Implementing Ball Tree algorithm as an optimization algorithm with the best value of K')
    print('\n')

    acc = 0
    model = KNeighborsClassifier(n_neighbors=maxK, algorithm='auto', leaf_size=100, metric='euclidean')
    model.fit(df1, Y_train)

    labels_train = Y_train

    pred = model.predict(df1)

    acc_train.append(metrics.accuracy_score(pred, labels_train))
    prec_train.append(metrics.precision_score(pred, labels_train))
    recal_train.append(metrics.recall_score(pred, labels_train))
    f1_train.append(metrics.f1_score(pred, labels_train))

    features_test = df2
    labels_test = Y_test

    pred = model.predict(features_test)

    acc_test.append(metrics.accuracy_score(pred, labels_test))
    prec_test.append(metrics.precision_score(pred, labels_test))
    recal_test.append(metrics.recall_score(pred, labels_test))
    f1_test.append(metrics.f1_score(pred, labels_test))

    acc = metrics.accuracy_score(pred, labels_test)
    acc = acc * 100
    print('Accuracy when K is at its best : ' + repr(acc))
    print('\n')

    ##################################Ball Tree#########################################################################

    acc = 0
    model = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree', leaf_size=150, metric='euclidean')
    model.fit(df1, Y_train)

    labels_train = Y_train

    pred = model.predict(df1)

    acc_train.append(metrics.accuracy_score(pred, labels_train))
    prec_train.append(metrics.precision_score(pred, labels_train))
    recal_train.append(metrics.recall_score(pred, labels_train))
    f1_train.append(metrics.f1_score(pred, labels_train))

    features_test = df2
    labels_test = Y_test

    pred = model.predict(features_test)

    acc_test.append(metrics.accuracy_score(pred, labels_test))
    prec_test.append(metrics.precision_score(pred, labels_test))
    recal_test.append(metrics.recall_score(pred, labels_test))
    f1_test.append(metrics.f1_score(pred, labels_test))

    acc = metrics.accuracy_score(pred, labels_test)
    acc = acc * 100
    print('Accuracy when K = 3 and Ball Tree algorithm is used : ' + repr(acc))
    print('\n')

    ####################################################KDTree##########################################################

    acc = 0
    model = KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree', leaf_size=200, metric='euclidean')
    model.fit(df1, Y_train)

    labels_train = Y_train

    pred = model.predict(df1)

    acc_train.append(metrics.accuracy_score(pred, labels_train))
    prec_train.append(metrics.precision_score(pred, labels_train))
    recal_train.append(metrics.recall_score(pred, labels_train))
    f1_train.append(metrics.f1_score(pred, labels_train))

    features_test = df2
    labels_test = Y_test

    pred = model.predict(features_test)

    acc_test.append(metrics.accuracy_score(pred, labels_test))
    prec_test.append(metrics.precision_score(pred, labels_test))
    recal_test.append(metrics.recall_score(pred, labels_test))
    f1_test.append(metrics.f1_score(pred, labels_test))

    acc = metrics.accuracy_score(pred, labels_test)
    acc = acc * 100
    print('Accuracy when K = 3 and KDTree algorithm is used ' + repr(acc))

    print("\n")
    print("Displaying Results for Training Set:")
    print("Accuracy : {}".format(acc_train))
    print("Precision : {}".format(prec_train))
    print("Recall : {}".format(recal_train))
    print("F1 : {}".format(f1_train))
    print("\n")
    print("Displaying Results for Test Set:")
    print("Accuracy : {}".format(acc_test))
    print("Precision : {}".format(prec_test))
    print("Recall : {}".format(recal_test))
    print("F1 : {}".format(f1_test))

    plotting1(k_range, k_accuracies)
    plotting2(df, train_label, 30)

if __name__ == '__main__':
    main()