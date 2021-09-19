import numpy as np
import sklearn as sk
import pandas as pd
from itertools import chain
from sklearn import  tree
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.model_selection import GridSearchCV
def read_csv(filename):
    # 数据为数字时可以，不可以时其他字符
    x_test=np.loadtxt(open(filename[0],"r"),delimiter=",",skiprows=0)
    list_test=str(x_test.tolist())
    print(x_test.shape)

    # print(x_test.shape)
    x_train = np.loadtxt(open(filename[1], "r"), delimiter=",", skiprows=0)
    x_val = np.loadtxt(open(filename[2], "r"), delimiter=",", skiprows=0)
    y_train = np.loadtxt(open(filename[3], "r"), delimiter=",", skiprows=0)
    y_val = np.loadtxt(open(filename[4], "r"), delimiter=",", skiprows=0)
    #删掉0行f1 减少了
    # x_index=np.where(x_train==0)
    # x_train=np.delete(x_train,x_index,axis=0)
    # y_train=np.delete(y_train,x_index,axis=0)
    # print(len(x_test),len(x_train),len(x_val),len(y_train),len(y_val))
    isempty=pd.DataFrame(x_train).isnull().sum()
    print(np.where(np.array(isempty)!=0))
    #数据中没有为0的列
    # 异常值检测不知道负的是否需要去除
    #离群值没有判断
    # print(x_train[:,1].shape)
    # exit()


    # out_x=outpoint["fliers"][0].get_xdata()
    # print(out_x)
    # print(np.argwhere(x_train[:,0] == int(list_outpoint[i]) for i in range(len(list_outpoint))))
    # index=np.zeros(len(list_outpoint))
    # print(index[0])
    index=[]
    p_ytrain=pd.DataFrame(y_train)
    p_ytrain=p_ytrain.rename(columns={0:"class"})
    sns.catplot(x="class", kind="count", data=p_ytrain)
    # print(chain.from_iterable(list(y_train[:,1])).count())
    plt.savefig("class distributions")
    plt.show()

    print(x_train.shape)
    # exit()
    x_train=x_train[:,[0,1,8,9,10,11,12,13,15,16,17,25,32,33,35,40,41,43,64,65,67,68,72,73,75,76,77,78,81,83,96,97,99,100,104,105,113,115,116,123,124]]
    x_val = x_val[:,
              [0, 1, 8, 9, 10, 11, 12, 13, 15, 16, 17, 25, 32, 33, 35, 40, 41, 43, 64, 65, 67, 68, 72, 73, 75, 76, 77,
               78, 81, 83, 96, 97, 99, 100, 104, 105, 113, 115, 116, 123, 124]]
    p_x = pd.DataFrame(x_train)
    plt.figure(figsize=(8, 6))
    outpoint=p_x.boxplot(return_type="dict")
    for col in range(x_train.shape[1]):
        out_val = outpoint["fliers"][col].get_ydata()
        print(len(out_val))
        list_outpoint = list(out_val)
        if len(list_outpoint)<0.015*x_train.shape[0]:
            for i in range(len(list_outpoint)):
                index.append(list(np.where(x_train[:, col] == list_outpoint[i])[0]))

    index_list = list(chain.from_iterable(index))
    print(len(set(index_list)))        #离群值共2949条数据
    x_train=np.delete(x_train,index_list,axis=0)
    y_train=np.delete(y_train,index_list,axis=0)
    # print(len(list_outpoint))
    print(x_train.shape)
    plt.savefig("f_b")
    plt.show()
    # for i in range(x_train.shape[1]):
    #     t_mean = np.mean(x_train[:,i])
    #     sigma=np.std(x_train)
    #     remove_out=np.where(abs(x_train[:,i]-t_mean)>5*sigma)
    #     print(remove_out)
    #     # x_train=np.delete(x_train,remove_out)
    # # print(x_train[:,i].shape)
    # exit()

    return x_test,x_train,x_val,y_train,y_val
    # print(data.shape)


def normalization(data_attrs):
    max_attr=np.max(data_attrs,axis=0)
    min_attr=np.min(data_attrs,axis=0)
    mean_attr=np.mean(data_attrs,axis=0)
    for i in range(data_attrs.shape[0]):
        for j in range(data_attrs.shape[1]):
            data_attrs[i][j]=(data_attrs[i][j]-mean_attr[j])/(max_attr[j]-min_attr[j])
    print("x_test",data_attrs)
    return data_attrs




def dt(x_train,x_test,x_val,y_train,y_val):
    model=tree.DecisionTreeClassifier(max_depth=30,min_samples_leaf=1,criterion="entropy")
    model.fit(x_train,y_train)
    y_pre=model.predict(x_val)
    tree.plot_tree(model)
    plt.savefig("dt_fig")
    plt.show()
    return y_pre


def knn(x_train,x_test,x_val,y_train,y_val):
    knn_clf=KNeighborsClassifier(n_neighbors=7,leaf_size=30)
    knn_clf.fit(x_train,y_train)
    y_pre=knn_clf.predict(x_val)
    probility_knn=knn_clf.predict_proba(x_val)  #计算各测试样本基于概率的预测
    nei_point=knn_clf.kneighbors(x_val[:-1],3,False)   #计算与最后一个测试样本距离在最近的3个点，返回的是这些样本的序号组成的数组
    print("probility",probility_knn)
    print("the final point :",nei_point)
    return y_pre



def seabornfig(x_train):
    sort_arr=np.array(pd.DataFrame(x_train).corr())[1:,1:]
    sort_arr=sort_arr[sort_arr<1.0]
    print(sort_arr)
    print(np.sort(sort_arr)[-10:])#argsort下标 ,sort值
    # print("Feature correlation coefficient:",np.max(np.where(np.array(pd.DataFrame(x_train).corr())[1:,1:]<1.0)))
    # print(pd.DataFrame(x_train).corr()[pd.DataFrame(x_train).corr()<0.5])
    sns.heatmap(pd.DataFrame(x_train).corr(),annot=False,cmap="OrRd")
    plt.savefig("feature_map")
    # sns.heatmap(pd.DataFrame(x_train[:3, :3]).corr(), annot=True, cmap="OrRd")
    plt.show()





def f1_score(y_val,prediction):

    f1_value_macro=sklearn.metrics.f1_score(y_val,prediction,average="macro")
    f1_value_micro=sklearn.metrics.f1_score(y_val,prediction,average="micro")
    f1_value_weightes=sklearn.metrics.f1_score(y_val,prediction,average="weighted")
    print(sklearn.metrics.classification_report(y_val,prediction))
    return f1_value_macro,f1_value_micro,f1_value_weightes

if __name__ == '__main__':
    datastr=["X_test.csv","X_train.csv","X_val.csv","y_train.csv","y_val.csv"]
    x_test,x_train,x_val,y_train,y_val=read_csv(datastr)
    p_xtrain=pd.DataFrame(x_train)
    p_ytrain=pd.DataFrame(y_train)
    p_ytrain=p_ytrain.rename(columns={0:42})
    p_x=pd.concat([p_xtrain,p_ytrain],axis=1,ignore_index=False)

    #for i,ax_i in enumerate(ax.flat):
    # for i in range(128):
    #     plt.scatter(p_x[128],p_x[i])
    #     # if i == 127:
    #     #     break
    #     plt.show()
    # print(p_x.shape)
    # exit()
    # print(p_x[42]) 121-127
    # # exit()
    # for i in range(p_x.shape[1]):
    #     if abs(p_x[i].corr(p_x[42],method="spearman"))<0.1:
    #         print(i)
    # sns.catplot(x=128, kind="count", data=p_ytrain)
    # plt.show()
    seabornfig(x_train)
    # exit()
    # dt_pre = dt(x_train, x_test, x_val, y_train, y_val)
    # x_train=normalization(x_train)
    # x_val=normalization(x_val)
    # x_test=normalization(x_test)
    # f1_dtval=f1_score(y_val,dt_pre)
    knn_pre=knn(x_train,x_test,x_val,y_train,y_val)
    f1_knnval=f1_score(y_val,knn_pre)
    # print("DT model:",f1_dtval[0],"\t",f1_dtval[1],"\t",f1_dtval[2])
    print("KNN model:",f1_knnval[0],"\t",f1_knnval[1],"\t",f1_knnval[2])

    # read_csv("test_d.tsv")