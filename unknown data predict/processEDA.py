"""
     comp 9417 group project
     Author: Na Wei
     Date 23/04/2021
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier




# 1.read "X_test.csv", "X_train.csv", "X_val.csv", "y_train.csv", "y_val.csv"
# 2.Perform data processing on the features filtered through the analysis of related variables (remove null values and outliers)
def read_csv(filename):
    x_test = np.loadtxt(open(filename[0], "r"), delimiter=",", skiprows=0)
    x_train = np.loadtxt(open(filename[1], "r"), delimiter=",", skiprows=0)
    x_val = np.loadtxt(open(filename[2], "r"), delimiter=",", skiprows=0)
    y_train = np.loadtxt(open(filename[3], "r"), delimiter=",", skiprows=0)
    y_val = np.loadtxt(open(filename[4], "r"), delimiter=",", skiprows=0)
    isempty = pd.DataFrame(x_train).isnull().sum()                           # Check whether the data has null values
    # print(isempty)                                                         #Data has no null values
    index = []
    p_ytrain = pd.DataFrame(y_train)
    p_ytrain = p_ytrain.rename(columns={0: "class"})
    sns.catplot(x="class", kind="count", data=p_ytrain)                      # show class distribution
    plt.show()
    x_train = x_train[:,
              [0, 1, 8, 9, 10, 11, 12, 13, 15, 16, 17, 25, 32, 33, 35, 40, 41, 43, 64, 65, 67, 68, 72, 73, 75, 76, 77,
               78, 81, 83, 96, 97, 99, 100, 104, 105, 113, 115, 116, 123, 124]]               #Index of relevant features
    x_val = x_val[:,
            [0, 1, 8, 9, 10, 11, 12, 13, 15, 16, 17, 25, 32, 33, 35, 40, 41, 43, 64, 65, 67, 68, 72, 73, 75, 76, 77,
             78, 81, 83, 96, 97, 99, 100, 104, 105, 113, 115, 116, 123, 124]]
    p_x = pd.DataFrame(x_train)
    plt.figure(figsize=(8, 6))
    outpoint = p_x.boxplot(return_type="dict")                                        #View the box plot of the data
    for col in range(x_train.shape[1]):
        out_val = outpoint["fliers"][col].get_ydata()
        list_outpoint = list(out_val)
        if len(list_outpoint) < 0.015 * x_train.shape[0]:                             #Choose a suitable threshold for the number of outliers to be deleted
            for i in range(len(list_outpoint)):
                index.append(list(np.where(x_train[:, col] == list_outpoint[i])[0]))

    index_list = list(chain.from_iterable(index))
    x_train = np.delete(x_train, index_list, axis=0)                                  # Remove outliers
    y_train = np.delete(y_train, index_list, axis=0)
    plt.show()
    return x_test, x_train, x_val, y_train, y_val



# show heatmap of relevant features
def seabornfig(x_train):
    sort_arr=np.array(pd.DataFrame(x_train).corr())[1:,1:]
    sort_arr=sort_arr[sort_arr<1.0]
    print("Top ten largest correlation coefficients:",np.sort(sort_arr)[-10:])                      # show top ten largest correlation coefficients
    sns.heatmap(pd.DataFrame(x_train).corr(),annot=False,cmap="OrRd")
    plt.show()


# knn classifier the appropriate k is 3
def knn(x_train,x_val,y_train):
    knn_clf=KNeighborsClassifier(n_neighbors=3,leaf_size=30)
    knn_clf.fit(x_train,y_train)
    y_pre=knn_clf.predict(x_val)
    return y_pre



# Evaluation data
def f1_score(y_val,prediction):
    f1_value_macro=sklearn.metrics.f1_score(y_val,prediction,average="macro")
    f1_value_micro=sklearn.metrics.f1_score(y_val,prediction,average="micro")
    f1_value_weightes=sklearn.metrics.f1_score(y_val,prediction,average="weighted")
    print(np.array(sklearn.metrics.classification_report(y_val,prediction)))
    return f1_value_macro,f1_value_micro,f1_value_weightes


if __name__ == '__main__':
    datastr = ["X_test.csv", "X_train.csv", "X_val.csv", "y_train.csv", "y_val.csv"]
    x_test, x_train, x_val, y_train, y_val = read_csv(datastr)
    p_xtrain = pd.DataFrame(x_train)
    p_ytrain = pd.DataFrame(y_train)
    p_ytrain = p_ytrain.rename(columns={0: 128})
    p_x = pd.concat([p_xtrain, p_ytrain], axis=1, ignore_index=False)
    seabornfig(x_train)
    knn_pre=knn(x_train,x_val,y_train)
    f1_knnval=f1_score(y_val,knn_pre)
