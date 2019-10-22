import os
import numpy as np
#import keras
from tqdm import tqdm
#import pandas as pd

#방식: 전체 데이터를 로드해서 섞고, 일부 배치를 제너레이팅하여 학습.
# image batch[n,144,144], label batch[n,10]
def load_dataset():
    data=[]
    label=[]
    dir='frames_npy'
    list=[[0,200],[315,615],[735,1056],[1155,1900],[1974,2358],[2520,2715],[2772,3576],[3654,4584],[4767,7520],[8169,10814]]
    fnames=os.listdir(dir)
    for i in tqdm(enumerate(list)):
        print(i) # [0,[0,200]]
        for j in range(i[1][0],i[1][1],1):
            fpath=os.path.join(dir,str(j).zfill(5)+'.npy')
            feature=np.load(fpath)
            #print(feature.shape) #(144,144)

            data.append(feature)
            label.append(i)

    data=np.array(data)
    label=np.array(label)

    data=data[ :, :, :, np.newaxis]
    #label= label[:, :, :, np.newaxis]
    print("data load done")
    return data,data

np.random.seed(123)


def load_data():
    #MNIST_M = np.load(data_path)
    train_data, train_label = load_dataset()
    #valid_data, valid_label = MNIST_M[1]
    #test_data, test_label = MNIST_M[2]

    return train_data, train_label


def batch_generator(X, y, batch_size, num_epochs, shuffle=True):
    data_size = X.shape[0]
    num_batches_per_epoch = data_size // batch_size + 1

    for epoch in range(num_epochs):
        # print("In epoch >> " + str(epoch + 1))
        # print("num batches per epoch is: " + str(num_batches_per_epoch))

        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            X_shuffled = X[shuffle_indices]
            y_shuffled = y[shuffle_indices]
        else:
            X_shuffled = X
            y_shuffled = y

        for batch_num in range(num_batches_per_epoch - 1):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            X_batch = X_shuffled[start_index:end_index]
            y_batch = y_shuffled[start_index:end_index]
            batch = list(zip(X_batch, y_batch))

            yield batch
