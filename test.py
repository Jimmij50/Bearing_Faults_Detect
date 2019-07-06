import pandas as pd
import numpy as np
import math
BATCH_SIZE=1
TRAIN_PATH='feature_train_e.csv'
TEST_PATH='feature_test_e.csv'
LENS=640
Data1=pd.read_csv(TRAIN_PATH)
TIME_PERIODS=33
def one_hot(label,classnum):
    height=label.shape[0]
    offset=np.arange(height)*classnum
    label=label.astype(int)
    onehot=np.zeros((height,classnum))
    onehot.flat[offset+label.ravel()]=1# 主要一定要吧label 变成整形
    return  onehot
def train_data(train):
    data=Data
    data=np.array(data)

    if train:
        data=data[:LENS]
    else:
        data=data[LENS:]
   #print(data.shape)
    steps=math.ceil(len(data)/BATCH_SIZE)
    np.random.shuffle(data)
    #print(data.shape)

    for i in range(steps):
        Batch=data[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]
        feature=Batch[:,1:-1]
        feature2 = []
        for j in feature:
            j=np.fft.fft(j)
            feature2.append(j)
        label=Batch[:,-1]
        label_onehot=one_hot(label,10)
        feature=np.reshape(feature,(-1,1,TIME_PERIODS,1))
        #label_onehot=np.reshape(label_onehot,(BATCH_SIZE,1,CLASS,1))
        yield feature	,label_onehot
t=train_data(True)
TEST_DIR=TEST_PATH='feature_test_e.csv'
#TEST_DIR=TEST_PATH='train.csv'
Data1=pd.read_csv(TRAIN_PATH)
Data1=np.array(Data1)
Data1=np.transpose(Data1)
data1=Data1[:,1:]
print(data1[0])
print(data1.shape)
Data=pd.read_csv(TEST_DIR)
Data=np.array(Data)
data=Data.transpose()
print(data[0])
print(data.shape)
feature=data
feature2 = []
for j in feature:
    # k=np.fft.fft(j)
    # feature2.append(abs(k))
    feature2.append(j)
feature2=np.array(feature2)
feature2=np.reshape(feature2,(-1,1,TIME_PERIODS,1))
