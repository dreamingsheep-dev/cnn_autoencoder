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
        #print(i) # [0,[0,200]]
        for j in range(i[1][0],i[1][1],1):
            fpath=os.path.join(dir,str(j).zfill(5)+'.npy')
            feature=np.load(fpath)
            #print(feature.shape) #(144,144)

            data.append(feature)
            label.append(i)

    data=np.array(data)
    label=np.array(label)

    return data,label

data,label=load_dataset()
print(data.shape) # (9277,144,144)
print(label.shape) # (9277,2)
