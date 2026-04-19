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

from keras.callbacks import EarlyStopping

device_lib.list_local_devices()
tf.config.list_physical_devices("GPU")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(406)
tf.random.set_seed(406)


# voxel_data = np.load('Overhang_voxel_0216.npy')
#
# print(np.shape(voxel_data))
#
# voxel_data_bool = voxel_data.astype('bool')

###########################빈 공간 데이터 확인(64*64*64)############################################
# inv_voxel_data = ~voxel_data_bool
#
# inv_voxel_data_del0 = np.delete(inv_voxel_data[0],0,axis=0)
# inv_voxel_data_del0 = np.delete(inv_voxel_data_del0,0,axis=1)
# inv_voxel_data_del0 = np.delete(inv_voxel_data_del0,0,axis=2)
# print(np.shape(inv_voxel_data_del0))
#
# inv_voxel_data_del1 = np.delete(inv_voxel_data_del0,64,axis=0)
# inv_voxel_data_del1 = np.delete(inv_voxel_data_del1,64,axis=1)
# inv_voxel_data_del1 = np.delete(inv_voxel_data_del1,64,axis=2)
# print(np.shape(inv_voxel_data_del1))
#############################################################################################




#####################################################################################################
####################################Voxelization#####################################################

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

# if __name__ == '__main__':
#
#     three_feature_app = natsort.natsorted(glob.glob("C:/work2/one_overhang_length&angle/A&A/*.stl"))
#     three_feature_app.extend(natsort.natsorted(glob.glob("C:/work2/one_overhang_length&angle_2/A&A/*.stl")))
# #
#     three_feature_inapp = natsort.natsorted(glob.glob("C:/work2/one_overhang_length&angle/F&F/*.stl"))
#     three_feature_inapp.extend(natsort.natsorted(glob.glob("C:/work2/one_overhang_length&angle_2/F&F/*.stl")))
# #
#     two_feature_app = natsort.natsorted(glob.glob("C:/work2/one_overhang_length&angle/overhang_A/*.stl"))
#     two_feature_app.extend(natsort.natsorted(glob.glob("C:/work2/one_overhang_length&angle_2/overhang_A/*.stl")))
#     # two_feature_app.extend(natsort.natsorted(glob.glob("C:/work/left_overhang_bl_Accept/*.stl")))
#     # two_feature_app.extend(natsort.natsorted(glob.glob("C:/work/left_overhang_bl_Accept_2/*.stl")))
#     # two_feature_app.extend(natsort.natsorted(glob.glob("C:/work/right_overhang_bl_Accept/*.stl")))
#     # two_feature_app.extend(natsort.natsorted(glob.glob("C:/work/right_overhang_bl_Accept_2/*.stl")))
#
# ########################################################################################################################################################################
#     one_feature_app = natsort.natsorted(glob.glob("C:/work2/one_overhang_length&angle/length_A/*.stl"))
#     one_feature_app.extend(natsort.natsorted(glob.glob("C:/work2/one_overhang_length&angle_2/length_A/*.stl")))
#     # one_feature_app.extend(natsort.natsorted(glob.glob("C:/work/right_overhang_Accept/*.stl")))
#     # one_feature_app.extend(natsort.natsorted(glob.glob("C:/work/right_overhang_Accept_2/*.stl")))
#     # one_feature_app.extend(natsort.natsorted(glob.glob("C:/work/bl_Accept/*.stl")))
#     # one_feature_app.extend(natsort.natsorted(glob.glob("C:/work/bl_Accept_2/*.stl")))
#
#     ## ############### ################ #######################################################################################################
#
#     all_hole_list = []
#     all_hole_list.extend(three_feature_app)
#     all_hole_list.extend(two_feature_app)
#     all_hole_list.extend(one_feature_app)
#     all_hole_list.extend(three_feature_inapp)
#
#     print(len(all_hole_list))
#
#     start = time.time()
#
#     voxel=[]
#
#     for input in all_hole_list:
#        vol_resolution = (65,65,65) # Resolution, into how many layers the model should be divided
#        cube_vol= stl_voxelizer(input, vol_resolution)
#        voxel.append(cube_vol)
#        # np.save(f'{all_hole_list.index(input)}.npy', cube_vol)
#        print("Current index = ", all_hole_list.index(input))
#
#
#     np.array(voxel)
#
#     np.save('one_overhang_length & angle_0108.npy',voxel) # Save voxel data
#
#     now = time.time()
#     print("CPU time = ", now - start)

# #
# #
##############################################Data slicing###################################################################
# origin_voxel = np.load("Wall_Thickness_Voxel_64.npy")
# split_index = origin_voxel.shape[0] // 2
# split_index_2 = origin_voxel.shape[0] // 16
#
# origin_voxel_A = origin_voxel[:split_index]
# origin_voxel_F = origin_voxel[split_index:]

# print(np.shape(origin_voxel_A))
# print(np.shape(origin_voxel_F))

# ex_voxel = np.load('Hole_0525.npy')
# ex_voxel_A = ex_voxel[:split_index]
# ex_voxel_F = ex_voxel[split_index:]
#
# print(np.shape(ex_voxel_A))
# print(np.shape(ex_voxel_F))

# split_voxel_D1 = origin_voxel_A[:split_index_2]
# split_voxel_D2 = origin_voxel_A[split_index_2*2:split_index_2*3]
# split_voxel_D3 = origin_voxel_A[split_index_2*4:split_index_2*5]
# split_voxel_D4 = origin_voxel_A[split_index_2*6:split_index_2*7]
#
# print(np.shape(split_voxel_D1))
# print(np.shape(split_voxel_D2))
# print(np.shape(split_voxel_D3))
# print(np.shape(split_voxel_D4))

# split_voxel_U1 = origin_voxel_A[split_index_2:split_index_2*2]
# split_voxel_U2 = origin_voxel_A[split_index_2*3:split_index_2*4]
# split_voxel_U3 = origin_voxel_A[split_index_2*5:split_index_2*6]
# split_voxel_U4 = origin_voxel_A[split_index_2*7:split_index_2*8]
#
# print(np.shape(split_voxel_U1))
# print(np.shape(split_voxel_U2))
# print(np.shape(split_voxel_U3))
# print(np.shape(split_voxel_U4))
#
# split_voxel_U1_F = origin_voxel_F[split_index_2:split_index_2*2]
# split_voxel_U2_F = origin_voxel_F[split_index_2*3:split_index_2*4]
# split_voxel_U3_F = origin_voxel_F[split_index_2*5:split_index_2*6]
# split_voxel_U4_F = origin_voxel_F[split_index_2*7:split_index_2*8]
#
# print(np.shape(split_voxel_U1_F))
# print(np.shape(split_voxel_U2_F))
# print(np.shape(split_voxel_U3_F))
# print(np.shape(split_voxel_U4_F))
#
# voxel_extend_list = []
# # voxel_extend_list = []
# #
# voxel_extend_list.extend(split_voxel_U1)
# voxel_extend_list.extend(split_voxel_U2)
# voxel_extend_list.extend(split_voxel_U3)
# voxel_extend_list.extend(split_voxel_U4)
#
# voxel_extend_list.extend(split_voxel_U1_F)
# voxel_extend_list.extend(split_voxel_U2_F)
# voxel_extend_list.extend(split_voxel_U3_F)
# voxel_extend_list.extend(split_voxel_U4_F)
#
# print(np.shape(voxel_extend_list))
# # print(np.shape(voxel_extend_list_F))
# #
# # voxel_extend_list.extend(voxel_extend_list_A)
# # voxel_extend_list.extend(voxel_extend_list_F)
# #
# # print(np.shape(voxel_extend_list))
# #
# np.save('wall_thickness_re_1123.npy',voxel_extend_list)



# ##################################### Multi_feature_test_sample ##############################################

    # cube = glob.glob("C:/Users/user/PycharmProjects/Feature_Recognition/완료/ValidDesign/S28.stl")
    # all_hole_list = []
    # all_hole_list.extend(cube)
    #
    # voxel=[]
    #
    # for input in all_hole_list:
    #     vol_resolution = (65,65,65) # Resolution, into how many layers the model should be divided #### 10:65, 20:129, 30:193, 40:257, 50: 321
    #     cube_vol= stl_voxelizer(input, vol_resolution)
    #     voxel.append(cube_vol)
    #     print("Current index = ", all_hole_list.index(input), "voxel=",cube_vol)
    #
    # np.array(voxel)
    #
    # np.save('C:/Users/user/PycharmProjects/Feature_Recognition/완료/ValidDesign/S28.npy',voxel)


############################################voxel 파일 순서, stl파일 순서 비교#############################################
#################### Define voxelization#########################

# def stl_voxelizer(file_names, Resolution):
#
#     input = file_names
#     meshes = []
#     resolution = Resolution
#     mesh_obj = mesh.Mesh.from_file(input)
#     org_mesh = np.hstack((mesh_obj.v0[:, np.newaxis], mesh_obj.v1[:, np.newaxis], mesh_obj.v2[:, np.newaxis]))
#     meshes.append(org_mesh)
#     vol, scale, shift = stltovoxel.convert_meshes(meshes,resolution)
#     # vol = np.rot90(vol, k=3, axes=(1,2))
#     return(vol)


##################################################################################################
##############################Data Preprocessing_(64*64*64)#######################################
# voxel_data_pp = np.load('C:/Users/user/PycharmProjects/Feature_Recognition/완료/Voxel/Finished_NPY/one_overhang_length & angle_0108.npy')
# # print(voxel_data_pp[0])
# print(np.shape(voxel_data_pp))
# voxel_test_64 = []
#
# for i in range(len(voxel_data_pp)):
#     print(i)
#     voxel_data_del0 = np.delete(voxel_data_pp[i],0,axis=0)
#     voxel_data_del0 = np.delete(voxel_data_del0,0,axis=1)
#     voxel_data_del0 = np.delete(voxel_data_del0,0,axis=2)
#     # print(np.shape(voxel_data_del0))
#     voxel_data_del1 = np.delete(voxel_data_del0,64,axis=0)
#     voxel_data_del1 = np.delete(voxel_data_del1,64,axis=1)
#     voxel_data_del1 = np.delete(voxel_data_del1,64,axis=2)
#     # print(np.shape(voxel_data_del1))
#
#     rot_voxel_x = np.rot90(voxel_data_del1, 1, axes=(0, 2))
#     rot_voxel_z = np.rot90(rot_voxel_x, 2, axes=(0, 1))
#
#     voxel_test_64.append(rot_voxel_z)
#
# np.save('C:/Users/user/PycharmProjects/Feature_Recognition/완료/Voxel/Finished_NPY/one_overhang_length & angle_pp_0111.npy',voxel_test_64)



################################### Multie feature preprocesing ######################################################
voxel_data_pp = np.load('C:/Users/user/PycharmProjects/Feature_Recognition/완료/ValidDesign/S28.npy')
print(voxel_data_pp[0])
print(np.shape(voxel_data_pp[0]))
#
voxel_data_pp = voxel_data_pp.astype('bool')
voxel_test_128 = []
voxel_data_del0 = np.delete(voxel_data_pp[0], 0, axis=0)
voxel_data_del0 = np.delete(voxel_data_del0, 0, axis=1)
voxel_data_del0 = np.delete(voxel_data_del0, 0, axis=2)
# print(np.shape(voxel_data_del0))
voxel_data_del1 = np.delete(voxel_data_del0, 64, axis=0) #### Z축        #### 10:64, 20:128, 30:192, 40:256, 50: 320
voxel_data_del1 = np.delete(voxel_data_del1, 64, axis=1)  #### Y축
voxel_data_del1 = np.delete(voxel_data_del1, 64, axis=2) #### X축
# print(np.shape(voxel_data_del1))

rot_voxel_x = np.rot90(voxel_data_del1, 1, axes=(0, 2))
rot_voxel_z = np.rot90(rot_voxel_x, 2, axes=(0, 1))

voxel_test_128.append(rot_voxel_z)

print(np.shape(voxel_test_128))

np.save('C:/Users/user/PycharmProjects/Feature_Recognition/완료/ValidDesign/S28.npy',voxel_test_128)

# # ################################################################################
# # ###########################visualization#######################################
#
# voxel_data_pp = np.load('C:/Users/user/PycharmProjects/Feature_Recognition/완료/Voxel/Finished_NPY/two_ovehang & one_bridge_pp_0111')
# # print(type(voxel_data_pp[0]))
# # print(voxel_data_pp[0])
# print(np.shape(voxel_data_pp))
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.voxels(voxel_data_pp[5000])
# # # ax.voxels(voxel_data_del1)
# #
# plt.show()

###################################################################################################
###################################### 모델 생성 ####################################################

# n = 2000
# feature_list = []
# feature_list_AAA = [5 for i in range(n)]
# feature_list_AAF = [4 for i in range(n)]
# feature_list_AFA = [3 for i in range(2*n)]
# feature_list_AF = [2 for i in range(2*n)]
# feature_list_FF = [1 for i in range(n)]
# feature_list_FFF = [0 for i in range(n)]
#
# # feature_list_AAA = [3 for i in range(n)]
# # feature_list_AAF = [2 for i in range(n)]
# # feature_list_AFA = [1 for i in range(n)]
# # feature_list_FFF = [0 for i in range(n)]
#
# # feature_list_A = [1 for i in range(n)]
# # feature_list_F = [0 for i in range(n)]
#
# # feature_list.extend(feature_list_A)
# # feature_list.extend(feature_list_F)
#
# feature_list.extend(feature_list_AAA)
# feature_list.extend(feature_list_AAF)
# feature_list.extend(feature_list_AFA)
# feature_list.extend(feature_list_AF)
# feature_list.extend(feature_list_FF)
# feature_list.extend(feature_list_FFF)
#
# X = np.load('C:/Users/user/PycharmProjects/Feature_Recognition/완료/Voxel/Finished_NPY/two_ovehang & one_bridge_pp_0111.npy')
# y = np.array(feature_list)
# y = pd.get_dummies(y)
# print(y)
#
# print('shape of x=', np.shape(X))
# print('shape of y=', np.shape(y))
# #
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
#
# print(np.shape(X_train))
# #https://runebook.dev/ko/docs/tensorflow/keras/layers/conv3d
# #(filters, kernel size, strides, activation)
#
# X_train_with_channels = np.expand_dims(X_train, axis=-1)
# print(np.shape(X_train_with_channels))
# X_test_with_channels = np.expand_dims(X_test, axis=-1)
#
# model = Sequential()
# model.add(Conv3D(32, kernel_size=(7, 7, 7), strides=(2, 2, 2), activation='relu', input_shape=(64, 64, 64, 1)))
# model.add(Conv3D(32, kernel_size=(5, 5, 5), strides=(1, 1, 1), activation='relu'))
# model.add(Conv3D(64, kernel_size=(4, 4, 4), strides=(1, 1, 1), activation='relu'))
# model.add(Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu'))
# model.add(Conv3D(64, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu'))
# model.add(MaxPooling3D(pool_size=(2, 2, 2)))
# model.add(layers.Dropout(0.5))
# model.add(layers.Flatten())
# model.add(layers.Dense(128, 'relu'))
# model.add(layers.Dense(6, 'softmax'))
#
# model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
#
# model.summary()
#
# early_stopping = EarlyStopping()
#
# history = model.fit(X_train_with_channels, y_train, validation_data=(X_test_with_channels, y_test), batch_size=256, epochs=100)
#
# print(history.history.keys())
#
# # model plot
# # summarize history for accuracy
#
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#
# model.save('two_ovehang & one_bridge_0111_3.h5')