from stl import mesh
import stltovoxel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import time
import natsort
import re
import tensorflow as tf
from tensorflow.keras.layers import (Conv3D, MaxPooling3D)
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import os
from tensorflow.python.client import device_lib
from tensorflow.keras.models import load_model

from skimage import data, io, filters, measure
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

device_lib.list_local_devices()
tf.config.list_physical_devices("GPU")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

np.random.seed(406)
tf.random.set_seed(406)

######################################### Define a voxelization function ###########################################

def stl_voxelizer(file_names, Resolution):

    input = file_names
    meshes = []
    resolution = Resolution
    mesh_obj = mesh.Mesh.from_file(input)
    org_mesh = np.hstack((mesh_obj.v0[:, np.newaxis], mesh_obj.v1[:, np.newaxis], mesh_obj.v2[:, np.newaxis]))
    meshes.append(org_mesh)
    vol, scale, shift = stltovoxel.convert_meshes(meshes,resolution)
    # vol = np.rot90(vol, k=3, axes=(1,2))
    return(vol)

#################################### Open train dataset #####################################################################
#
# if __name__ == '__main__':
# #
# ############### Unacceptable dataset ##########################
#
# #     app_hole_list = glob.glob("C:/Users/user/Desktop/Feature_data/Overhang/Overhang_1/accept/*.stl")
# #     app_hole_list.extend(glob.glob("C:/Users/user/Desktop/Feature_data/Overhang/Overhang_2/accept/*.stl"))
# #     app_hole_list.extend(glob.glob("C:/Users/user/Desktop/Feature_data/Overhang/Overhang_3/accept/*.stl"))
# #     app_hole_list.extend(glob.glob("C:/Users/user/Desktop/Feature_data/Overhang/Overhang_4/accept/*.stl"))
# #
# #
# # ################# Acceptable dataset ##########################
# #
# #     inapp_hole_list = glob.glob("C:/Users/user/Desktop/Feature_data/Overhang/Overhang_1/fail/*.stl")
# #     inapp_hole_list.extend(glob.glob("C:/Users/user/Desktop/Feature_data/Overhang/Overhang_2/fail/*.stl"))
# #     inapp_hole_list.extend(glob.glob("C:/Users/user/Desktop/Feature_data/Overhang/Overhang_3/fail/*.stl"))
# #     inapp_hole_list.extend(glob.glob("C:/Users/user/Desktop/Feature_data/Overhang/Overhang_4/fail/*.stl"))
# #
# #     all_hole_list = []
# #     all_hole_list.extend(app_hole_list)
# #     all_hole_list.extend(inapp_hole_list)
#
# ##################################################################################################
#
# ####################### Voxelization ########################
#
#     cube = glob.glob("C:/Users/user/Desktop/Feature_data/overhang_AF.stl")
#     all_hole_list = []
#     all_hole_list.extend(cube)
#
#     voxel=[]
#
#     for input in all_hole_list:
#         vol_resolution = 129 # Resolution, into how many layers the model should be divided
#         cube_vol= stl_voxelizer(input, vol_resolution)
#         voxel.append(cube_vol)
#         print("Current index = ", all_hole_list.index(input), "voxel=",cube_vol)
#
#     np.array(voxel)
#
#     np.save('overhang_test_128.npy',voxel) # Save voxel data
#
#     start = time.time()
#     print("time:", time.time() - start)

######################### Voxelization_test dataset ########################
    #
    # test = glob.glob("C:/Users/user/PycharmProjects/Feature_Recognition/20.stl")
    # all_hole_list = []
    # all_hole_list.extend(test)
    #
    # voxel=[]
    #
    # for input in all_hole_list:
    #     vol_resolution = 129 # Resolution, into how many layers the model should be divided
    #     cube_vol = stl_voxelizer(input, vol_resolution)
    #     # cube_vol = np.pad(cube_vol, pad_width=((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)
    #     voxel.append(cube_vol)
    #     print(np.shape(cube_vol))
    #     print("Current index = ", all_hole_list.index(input), "voxel=",cube_vol)
    #
    # np.array(voxel)
    #
    # np.save('20.npy',voxel) # Save voxel data

################################### Open voxel dataset (Feature가 있는지 없는지 확인 해보기 위한 Data 전처리)###############################################################

# all_hole_list_voxel = np.load('overhang_voxel_pp.npy')
# cube_voxel = np.load('cube_voxel.npy')
# all_hole_list_voxel = all_hole_list_voxel.tolist()
# cube_voxel = cube_voxel.tolist()
#
# cube_voxel_list=[]
#
# n = 4000 # 전체 데이터 개수
# for i in range(n):
#     cube_voxel_list.extend(cube_voxel)
#     print(i)
#
# all_hole_list_voxel.extend(cube_voxel_list)
# all_hole_list_voxel = np.array(all_hole_list_voxel)
#
# np.save('hole_voxel_comp.npy',all_hole_list_voxel)
#
# feature_list = pd.get_dummies(feature_list)
#
# print(feature_list)
#
#
# feature_list_ohe=pd.get_dummies(feature_list)
#
#
# feature_list = np.array(feature_list)
#
# print(feature_list.reshape((8, 1)))
#
# print(feature_list)


# ###################################### Split the datasets ####################################################
#
# all_hole_list_voxel = np.load('C:/Users/user/Desktop/희나/FR/overhang_voxel.npy')
# # # all_hole_list_voxel = np.load('hole_voxel_comp.npy')
#
#
# X = np.array(all_hole_list_voxel)
#
# print(np.shape(X))
# #
# # ################# Test 데이터셋 생성 ###################
# #
# a0 = np.array([X[100]])
# a1 = np.array([X[200]])
# a2 = np.array([X[300]])
# a3 = np.array([X[400]])
# a4 = np.array([X[500]])
# a5 = np.array([X[600]])
# a6 = np.array([X[700]])
# a7 = np.array([X[800]])
# a8 = np.array([X[900]])
# a9 = np.array([X[1000]])
# a10 = np.array([X[1100]])
# a11 = np.array([X[1200]])
# a12 = np.array([X[1300]])
# a13 = np.array([X[1400]])
# a14 = np.array([X[1500]])
# a15 = np.array([X[1600]])
# a16 = np.array([X[1700]])
# a17 = np.array([X[1800]])
# a18 = np.array([X[1900]])
# a19 = np.array([X[1999]])
#
#
# test = np.concatenate([a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19])
#
# X_list = X.tolist()
#
# del X_list[1999]
# del X_list[1900]
# del X_list[1800]
# del X_list[1700]
# del X_list[1600]
# del X_list[1500]
# del X_list[1400]
# del X_list[1300]
# del X_list[1200]
# del X_list[1100]
# del X_list[1000]
# del X_list[900]
# del X_list[800]
# del X_list[700]
# del X_list[600]
# del X_list[500]
# del X_list[400]
# del X_list[300]
# del X_list[200]
# del X_list[100]
#
#
# X = np.array(X_list)
# #
# np.save('train_overhang_voxel.npy',X)
# np.save('test_overhang_voxel.npy', test)


# ###################################### Developement of 3D-CNN model ####################################################

n = 2000 # 전체 데이터 개수의 절반
feature_list = [3 for i in range(n)]
feature_list_2 = [0 for i in range(n)]
feature_list_1 = [1 for i in range(n)]
feature_list_0 = [2 for i in range(n)]

# feature_list_AAA = [5 for i in range(n)]
# feature_list_AAF = [4 for i in range(n)]
# feature_list_AFA = [3 for i in range(2*n)]
# feature_list_AF = [2 for i in range(2*n)]
# feature_list_FF = [1 for i in range(n)]
# feature_list_FFF = [0 for i in range(n)]

# feature_list.extend(feature_list_4)
# feature_list.extend(feature_list_3)
# feature_list.extend(feature_list_2)

feature_list.extend(feature_list_2)
feature_list.extend(feature_list_1)
feature_list.extend(feature_list_0)

X = np.load('C:/Users/user/PycharmProjects/Feature_Recognition/완료/Voxel/Finished_NPY/one_overhang_length & angle_pp_0111.npy')
y = np.array(feature_list)
y = pd.get_dummies(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

X_train_with_channels = np.expand_dims(X_train, axis=-1)

print(np.shape(X_train_with_channels))
X_test_with_channels = np.expand_dims(X_test, axis=-1)
#https://runebook.dev/ko/docs/tensorflow/keras/layers/conv3d
#(filters, kernel size, strides, activation)

model = Sequential()
model.add(Conv3D(32, kernel_size=(7, 7, 7), strides=(2, 2, 2), activation='relu', input_shape=(64, 64, 64, 1)))
model.add(Conv3D(32, kernel_size=(5, 5, 5), strides=(1, 1, 1), activation='relu'))
model.add(Conv3D(64, kernel_size=(4, 4, 4), strides=(1, 1, 1), activation='relu'))
model.add(Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(128, 'relu'))
model.add(layers.Dense(4, 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

model.summary()

with tf.device('/device:GPU:0'):

    history = model.fit(X_train_with_channels, y_train, validation_data=(X_test_with_channels, y_test), batch_size = 128, epochs = 50)

    print(history.history.keys())
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.plot(history.history['accuracy'], color='tomato')
    plt.plot(history.history['val_accuracy'], color='royalblue')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy',fontsize=20)
    plt.xlabel('Epoch',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(['Train', 'Validation'], fontsize=20, loc='upper left')
    plt.show()

    # model.save('overhang_model_pp.h5')

######################################################################################################################

# test_data = np.load('overhang_voxel_test_pp.npy')
#
# ## test_model = load_model('through_hole_comp_model.h5')
#
# test_model = load_model('overhang_model_pp.h5')
#
# # print(test_data)
# for i in range(8):
#     a = np.array([test_data[i]])
#     # print(test_data[i])
#     # print(np.shape(a))
#     test_y = test_model.predict(a)
#     test_answer = test_y[0][0]
#     print(round(test_answer))
