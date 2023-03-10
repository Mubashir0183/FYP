from keras.models import model_from_json
import tensorflow as tf
import numpy as np
import sys
import os 
import cv2
import math
from heatmap import heatmap

dataset = 'B'
img_path = 'D:\FYP\MCNN\data/original/shanghaitech/part_' + dataset + '_final/test_data/images/'
den_path = 'D:\FYP\MCNN\data/original/shanghaitech/part_' + dataset + '_final/test_data/ground_truth_csv/'

def data_pre_test():
    print('loading test data from dataset', dataset, '...')
    img_names = os.listdir(img_path)
    img_names = img_names
    img_num = len(img_names)

    data = []
    for i in range(1, img_num + 1):
        print(i, '/', img_num)
        name = 'IMG_' + str(i) + '.jpg'
        img = cv2.imread(img_path + name, 0)
        img = np.array(img)
        img = (img - 127.5) / 128
        #print(img.shape)
        den = np.loadtxt(open(den_path + name[:-4] + '.csv'), delimiter = ",")
        #print(den.shape)
        den_sum = np.sum(den)
        data.append([img, den_sum])
            
    print('load data finished.')
    return data
    
data = data_pre_test()

model = model_from_json(open('Trained_Model/model.json').read())
model.load_weights('Trained_Model/weights.h5')

mae = 0
mse = 0
count = 1
accuracy = 0
img_names = os.listdir(img_path)
img_num = len(img_names)

for d in data:
    #for heatmap
    img_names = os.listdir(img_path)
    img_num = len(img_names)
    name = 'IMG_' + str(count) + '.jpg'
    den1 = np.loadtxt(open(den_path + name[:-4] + '.csv'), delimiter = ",")
    #######

    inputs = np.reshape(d[0], [1, 768, 1024, 1])
    outputs = model.predict(inputs)
    den = d[1]
    c_act = np.sum(den)
    c_pre = np.sum(outputs)
    print("image Number ",count)
    heatmap(den1, count, dataset, 'act')
    count=count+1
    print('predicted count:', c_pre)
    print('actual value:', c_act)
    if(c_act>c_pre):
        accuracy=c_pre/c_act
        accuracy=accuracy*100
    else:
        accuracy=c_act/c_pre
        accuracy=accuracy*100
    print("accuracy = ",accuracy)
    mae += abs(c_pre - c_act)
    mse += (c_pre - c_act) * (c_pre - c_act)
mae /= len(data)
mse /= len(data)
mse = math.sqrt(mse)

print('#############################')
print('#############################')
print('#############################')
print('#############################')
print('mae:', mae, 'mse:', mse)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
