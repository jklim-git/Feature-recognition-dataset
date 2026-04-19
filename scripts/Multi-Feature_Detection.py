import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import time
from keras.layers import (Conv3D, MaxPooling3D)
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import os
from tensorflow.python.client import device_lib
from keras.models import load_model
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from skimage import data, io, filters, measure
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

device_lib.list_local_devices()

start_time = time.time()

multi_features = np.load('C:/Users/user/PycharmProjects/Feature_Recognition/완료/ValidDesign/S24.npy')

multi_features = multi_features[0]
multi_features = multi_features.astype('bool')

print(np.shape(multi_features))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.voxels(multi_features)
plt.show()

############################### Split into subtractive and additive features  ###############################

cube_resolution = 64
cube_centrality = 32
real_resolution_x = np.shape(multi_features)[0]
real_resolution_y = np.shape(multi_features)[1]
real_resolution_z = np.shape(multi_features)[2]

real_resolution_index_x = real_resolution_x - 1
real_resolution_index_y = real_resolution_y - 1
real_resolution_index_z = real_resolution_z - 1
sub_point = real_resolution_z

for i in range(real_resolution_z):
    if multi_features[0][0][real_resolution_index_z - i] == True \
            and multi_features[0][real_resolution_index_y][real_resolution_index_z - i] == True \
            and multi_features[real_resolution_index_x][0][real_resolution_index_z - i] == True \
            and multi_features[real_resolution_index_x][real_resolution_index_y][real_resolution_index_z - i] == True:
        sub_point = real_resolution_index_z - i + 1
        break

sub_feat = multi_features[:,:,0:sub_point]
add_feat = multi_features[:,:,sub_point:real_resolution_z]

########### Remove noise ######################

for i in range(real_resolution_x):
    for j in range(sub_point):
        if sub_feat[i][0][j] == 0 and sub_feat[i][1][j] == 1:
            sub_feat[i][0][j] = 1

for i in range(real_resolution_x):
    for j in range(sub_point):
        if sub_feat[i][real_resolution_index_y][j] == 0 and sub_feat[i][real_resolution_index_y - 1][j] == 1:
            sub_feat[i][real_resolution_index_y][j] = 1

for i in range(real_resolution_y):
    for j in range(sub_point):
        if sub_feat[0][i][j] == 0 and sub_feat[0][i][j] == 1:
            sub_feat[0][i][j] = 1

for i in range(real_resolution_y):
    for j in range(sub_point):
        if sub_feat[real_resolution_index_x][i][j] == 0 and sub_feat[real_resolution_index_x - 1][i][j] == 1:
            sub_feat[real_resolution_index_x][i][j] = 1

c_1 = add_feat

final_labels_1 = np.zeros(c_1.shape)
all_c_1 = measure.label(c_1)

# print(c_1.shape)
# print(final_labels_1)
# print(all_c_1.shape)

b_2 = sub_feat
c_2 = ~b_2

final_labels_2 = np.zeros(c_2.shape)
all_c_2 = measure.label(c_2)

# fig = plt.figure()
# ax = fig.add_subplot(projection = '3d')
# ax.voxels(add_feat)
# plt.show()
#
def get_seg_samples_additive(seg_results,resolution):

    samples = np.zeros((0, resolution, resolution, resolution))

    max_dim = np.shape(seg_results)

    for i in range(1, np.max(seg_results.astype(int)) + 1):

        idx = np.where(seg_results == i)

        if len(idx[0]) == 0:
            continue

        cursample = np.zeros((resolution, resolution, resolution))

        for j in range(len(idx)):
            moving = 0
            max_length = np.max(idx[j]) #피쳐에 대한 한 축의 최대 위치
            min_length = np.min(idx[j]) #피쳐에 대한 한 축의 최소 위치
            central_move = 0
            centrality = round((max_length + min_length) / 2, 0)
            central_move = centrality - cube_centrality # 중앙이동
            width_axis = 0

            if j == len(idx) - 1:
                height = max_length

                if height >= 54:
                    height = height - 10

            if max_length - min_length < cube_resolution - 1 and j!=2:
                width_axis = 1

            if max_length >= 64:
                moving = max_length - 63  # 피쳐가 벗어난 만큼 움직이는 범위 계산

            for k in range(len(idx[j])):

                original = idx[j][k]

                if moving <= 0 and width_axis == 1:
                    idx[j][k] = original - central_move  ##### 나머지 피쳐 간격 움직임

                if moving > 0:
                    moved_coor = idx[j][k] - moving

                    if moved_coor >= 0:
                        idx[j][k] = moved_coor

                        if width_axis == 1:
                            idx[j][k] = original - central_move  ##### 나머지 피쳐 간격 움직임

                    if moved_coor < 0:
                        idx[j][k] = 1

        cursample[idx] = 1

        cursample[:,:,height:resolution] = 1
        cursample = np.flip(cursample, axis=2)
        cursample = np.expand_dims(cursample, axis=0)
        samples = np.append(samples, cursample, axis=0)

    return samples
# # # #
def get_seg_samples_subtractive(seg_results,resolution):

    samples = np.zeros((0, resolution, resolution, resolution))

    max_dim = np.shape(seg_results)

    for i in range(1, np.max(seg_results.astype(int)) + 1):

        idx = np.where(seg_results == i)

        if len(idx[0]) == 0:
            continue

        cursample = np.ones((resolution, resolution, resolution))

        width = 0
        height = 0
        angle_act = 0

        for j in range(len(idx)):

            if j == 0 :
                real_resolution = real_resolution_x

            if j == 1:
                real_resolution = real_resolution_y

            if j == 2:
                real_resolution = real_resolution_z

            width_axis = 0
            moving = 0
            max_length = np.max(idx[j]) #피쳐에 대한 한 축의 최대 위치
            min_length = np.min(idx[j]) #피쳐에 대한 한 축의 최소 위치
            centrality = round((max_length + min_length) / 2, 0)
            central_move = centrality - cube_centrality #중앙이동

            if max_length - min_length < cube_resolution - 1 and j!=2:
                width = max_length - min_length #피쳐에 너비 축 파악 및 계산
                width_axis = 1

                if min_length == 0 or max_length == real_resolution - 1:
                    angle_act = 1

            if j == 2 and min_length != 0:
                moving = min_length
                height = max_length - min_length #밑이 아닌 피쳐에 높이 계산

            if j != 2 and max_length >= 64:
                moving = max_length - 63 #피쳐가 벗어난 만큼 움직이는 범위 계산

            for k in range(len(idx[j])):

                original = idx[j][k]

                if moving <= 0:

                    if width_axis == 1 and min_length != 0 and max_length != real_resolution - 1:
                        idx[j][k] = original - central_move  #나머지 피쳐 간격 움직임````

                    if height != 0 and width != 0 and j == 2:
                        dev = abs(width - height)

                        if dev <= 1 and angle_act == 0:
                            idx[j][k] = original - central_move  #Hole diameter 높이 움직임

                if moving > 0:
                    moved_coor = idx[j][k] - moving

                    if moved_coor >= 0:
                        idx[j][k] = moved_coor

                        if height !=0 and width !=0 and j == 2:
                            dev = abs(width - height)

                            if dev <= 1 and angle_act == 0:
                                idx[j][k] = original - central_move  # Hole diameter 높이 움직임

                        if width_axis == 1 and max_length != real_resolution - 1 and min_length != 0:
                            idx[j][k] = original - central_move #나머지 피쳐 간격 움직임

                    if moved_coor < 0:
                        idx[j][k] = 0

        cursample[idx] = 0
        cursample = np.expand_dims(cursample, axis=0)
        samples = np.append(samples, cursample, axis=0)

    return samples
#
# ################################ Watershed Algorithm_Additive Features ###############################

for i in range(1, np.max(all_c_1) + 1):

    mk = (all_c_1 == i)  #### 따로 추출

    distance = ndi.distance_transform_edt(mk)

    labels = watershed(-distance)

    max_val = np.max(final_labels_1) + 1

    idx = np.where(mk)

    final_labels_1[idx] += (labels[idx] + max_val)
# # # #
# # # # ################################ Watershed Algorithm_subractive Features ###############################
# # #
for i in range(1, np.max(all_c_2) + 1):

    mk = (all_c_2 == i)  #### 따로 추출

    distance = ndi.distance_transform_edt(mk)

    labels = watershed(-distance)

    max_val = np.max(final_labels_2) + 1

    idx = np.where(mk)

    final_labels_2[idx] += (labels[idx] + max_val)
#
results_1 = get_seg_samples_additive(final_labels_1,cube_resolution)
results_2 = get_seg_samples_subtractive(final_labels_2,cube_resolution)

comp_results = np.concatenate((results_1, results_2))

print(np.shape(comp_results))
# #
# np.save('C:/Users/user/PycharmProjects/Feature_Recognition/완료/IndustrialPart/Support_comp.npy',comp_results)

# comp_results = np.load('C:/Users/user/PycharmProjects/Feature_Recognition/완료/CaseDesign/num8_comp.npy')
# print(len(comp_results))

# for i in range(len(comp_results)):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.voxels(comp_results[i]) # 29
#     plt.show()
    # plt.savefig('C:/Users/user/PycharmProjects/Feature_Recognition/완료/CaseDesign/no.5/'+str(i)+'.png')
    # plt.close()
# ######################################## Integrated 3D-CNN model for multi-design features evaluations #############################################

num_features = len(comp_results)
recognition_model = load_model('C:/Users/user/PycharmProjects/Feature_Recognition/완료/model/11Features_Recognition_0530.h5')

hole_evaluation_model = load_model('C:/Users/user/PycharmProjects/Feature_Recognition/완료/model/single/Hole_model_final.h5')
overhang_length_evaluation_model = load_model('C:/Users/user/PycharmProjects/Feature_Recognition/완료/model/overhang_2/Overhang_length_1004.h5')
overhang_angle_evaluation_model = load_model('C:/Users/user/PycharmProjects/Feature_Recognition/완료/model/single/Overhang_model_final.h5')
overhang_corner_evaluation_model = load_model('C:/Users/user/PycharmProjects/Feature_Recognition/완료/model/overhang_2/Overhang_corner_1004.h5')
overhang_inner_evaluation_model = load_model('C:/Users/user/PycharmProjects/Feature_Recognition/완료/model/overhang_2/Overhang_inner_1004.h5')
bridge_evaluation_model = load_model('C:/Users/user/PycharmProjects/Feature_Recognition/완료/model/single/Bridge_Length_0306.h5')
pin_evaluation_model = load_model('C:/Users/user/PycharmProjects/Feature_Recognition/완료/model/single/Pin_diameter_model_0224.h5')
wall_evaluation_model = load_model('C:/Users/user/PycharmProjects/Feature_Recognition/완료/model/single/Wall_thickness_model_final.h5')

two_overhang_evaluation_model = load_model('C:/Users/user/PycharmProjects/Feature_Recognition/완료/model/interacted/two_overhang_1006.h5')
one_overhang_length_angle_evaluation_model = load_model('C:/Users/user/PycharmProjects/Feature_Recognition/완료/model/interacted/one_overhang_length & angle_0111.h5')
two_overhang_bridge_evaluation_model = load_model('C:/Users/user/PycharmProjects/Feature_Recognition/완료/model/interacted/two_ovehang & one_bridge_0111_3.h5')

hole_count = 0
overhang_length_count = 0
overhang_angle_count = 0
overhang_corner_count = 0
overhang_inner_count = 0
bridge_count = 0
pin_count = 0
wall_count = 0

hole_accept = 0
hole_non_accept = 0
overhang_length_accept = 0
overhang_length_non_accept = 0
overhang_angle_accept = 0
overhang_angle_non_accept = 0
overhang_corner_accept = 0
overhang_corner_non_accept = 0
overhang_inner_accept = 0
overhang_inner_non_accept = 0
bridge_accept = 0
bridge_non_accept = 0
pin_accept = 0
pin_non_accept = 0
wall_accept = 0
wall_non_accept = 0

for i in range(num_features):
    answer_x = np.array([comp_results[i]])
    answer_x = np.expand_dims(answer_x, axis=-1)
    test_y = recognition_model.predict(answer_x)
    # print(test_y)

    feature_index = np.argmax(test_y[0])

    if feature_index == 0: # Through Hole
        hole_count = hole_count + 1
        evaluation_y = hole_evaluation_model.predict(answer_x)
        answer = round(evaluation_y[0][0])
        print("Hole Diameter")
        print(answer)

        if answer == 1:
            hole_accept = hole_accept + 1
            ans = str('Suitable')
        if answer == 0:
            hole_non_accept = hole_non_accept + 1
            ans = str('Non_Suitable')

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.voxels(comp_results[i])  # 29
        # plt.savefig('C:/Users/user/PycharmProjects/Feature_Recognition/완료/CaseDesign/no.8/' + str(i) + '.Hole Diameter_' + str(ans) +'.png')
        # plt.close()

    if feature_index == 1: # Overhang length
        overhang_length_count = overhang_length_count + 1
        evaluation_y = overhang_length_evaluation_model.predict(answer_x)
        answer = round(evaluation_y[0][0])
        print("Overhang Length")
        print(answer)

        if answer == 1:
            overhang_length_accept = overhang_length_accept + 1
            ans = str('Suitable')

        if answer == 0:
            overhang_length_non_accept = overhang_length_non_accept + 1
            ans = str('Non_Suitable')

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.voxels(comp_results[i])  # 29
        # plt.savefig('C:/Users/user/PycharmProjects/Feature_Recognition/완료/CaseDesign/no.8/' + str(i) + '.Overhang Length_' + str(ans) +'.png')
        # plt.close()

    if feature_index == 2: # Overhang angle
        overhang_angle_count = overhang_angle_count + 1
        evaluation_y = overhang_angle_evaluation_model.predict(answer_x)
        answer = round(evaluation_y[0][0])
        print("Overhang Angle")
        print(answer)

        if answer == 1:
            overhang_angle_accept = overhang_angle_accept + 1
            ans = str('Suitable')

        if answer == 0:
            overhang_angle_non_accept = overhang_angle_non_accept + 1
            ans = str('Non_Suitable')

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.voxels(comp_results[i])  # 29
        # plt.savefig('C:/Users/user/PycharmProjects/Feature_Recognition/완료/CaseDesign/no.8/' + str(i) + '.Overhang Angle_' + str(ans) +'.png')
        # plt.close()

    if feature_index == 3: # Overhang corner
        overhang_corner_count = overhang_corner_count + 1
        evaluation_y = overhang_corner_evaluation_model.predict(answer_x)
        answer = round(evaluation_y[0][0])
        print("Overhang Corner")
        print(answer)

        if answer == 1:
            overhang_corner_accept = overhang_corner_accept + 1
            ans = str('Suitable')

        if answer == 0:
            overhang_corner_non_accept = overhang_corner_non_accept + 1
            ans = str('Non_Suitable')

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.voxels(comp_results[i])  # 29
        # plt.savefig('C:/Users/user/PycharmProjects/Feature_Recognition/완료/CaseDesign/no.8/' + str(i) + '.Overhang Corner_' + str(ans) +'.png')
        # plt.close()

    if feature_index == 4: # Overhang inner
        overhang_inner_count = overhang_inner_count + 1
        evaluation_y = overhang_inner_evaluation_model.predict(answer_x)
        answer = round(evaluation_y[0][0])
        print("Overhang Inner")
        print(answer)

        if answer == 1:
            overhang_inner_accept = overhang_inner_accept + 1
            ans = str('Suitable')

        if answer == 0:
            overhang_inner_non_accept = overhang_inner_non_accept + 1
            ans = str('Non_Suitable')

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.voxels(comp_results[i])  # 29
        # plt.savefig('C:/Users/user/PycharmProjects/Feature_Recognition/완료/CaseDesign/no.8/' + str(i) + '.Overhang Inner_' + str(ans) +'.png')
        # plt.close()

    if feature_index == 5:  # Bridge
        bridge_count = bridge_count + 1
        evaluation_y = bridge_evaluation_model.predict(answer_x)
        answer = round(evaluation_y[0][0])
        print("Bridge Length")
        print(answer)

        if answer == 1:
            bridge_accept = bridge_accept + 1
            ans = str('Suitable')

        if answer == 0:
            bridge_non_accept = bridge_non_accept + 1
            ans = str('Non_Suitable')

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.voxels(comp_results[i])  # 29
        # plt.savefig('C:/Users/user/PycharmProjects/Feature_Recognition/완료/CaseDesign/no.8/' + str(i) + '.Bridge Length_' + str(ans) +'.png')
        # plt.close()

    if feature_index == 6: # Pin
        pin_count = pin_count + 1
        evaluation_y = pin_evaluation_model.predict(answer_x)
        answer = round(evaluation_y[0][0])
        print("Pin Diameter")
        print(answer)

        if answer == 1:
            pin_accept = pin_accept + 1
            ans = str('Suitable')

        if answer == 0:
            pin_non_accept = pin_non_accept + 1
            ans = str('Non_Suitable')

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.voxels(comp_results[i])  # 29
        # plt.savefig('C:/Users/user/PycharmProjects/Feature_Recognition/완료/CaseDesign/no.8/' + str(i) + '.Pin Diameter_' + str(ans) +'.png')
        # plt.close()

    if feature_index == 7: # Wall
        wall_count = wall_count + 1
        evaluation_y = wall_evaluation_model.predict(answer_x)
        answer = round(evaluation_y[0][0])
        print("Wall Thickness")
        print(answer)

        if answer == 1:
            wall_accept = wall_accept + 1
            ans = str('Suitable')

        if answer == 0:
            wall_non_accept = wall_non_accept + 1
            ans = str('Non_Suitable')

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.voxels(comp_results[i])  # 29
        # plt.savefig('C:/Users/user/PycharmProjects/Feature_Recognition/완료/CaseDesign/no.8/' + str(i) + '.Wall Thickness_' + str(ans) +'.png')
        # plt.close()

    if feature_index == 8: # Two overhang
        overhang_angle_count = overhang_angle_count + 2
        evaluation_y = two_overhang_evaluation_model.predict(answer_x)
        answer = np.argmax(evaluation_y[0])
        print("Two Overhang Angles")
        print(answer)

        if answer == 2:
            overhang_angle_accept = overhang_angle_accept + 2
            ans = str('All_Suitable')

        if answer == 1:
            overhang_angle_accept = overhang_angle_accept + 1
            overhang_angle_non_accept = overhang_angle_non_accept + 1
            ans = str('One OA_Suitable & One OA_Non_Suitable')

        if answer == 0:
            overhang_angle_non_accept = overhang_angle_non_accept + 2
            ans = str('All_Non_Suitable')

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.voxels(comp_results[i])  # 29
        # plt.savefig('C:/Users/user/PycharmProjects/Feature_Recognition/완료/CaseDesign/no.8/' + str(i) + '.Two Overhang Angles_' + str(ans) +'.png')
        # plt.close()

    if feature_index == 9:  # One overhang & length
        overhang_angle_count = overhang_angle_count + 1
        overhang_length_count = overhang_length_count + 1
        evaluation_y = one_overhang_length_angle_evaluation_model.predict(answer_x)
        answer = np.argmax(evaluation_y[0])
        print("One Overhang Angle & One Overhang Length")
        print(answer)

        if answer == 3:
            overhang_angle_accept = overhang_angle_accept + 1
            overhang_length_accept = overhang_length_accept + 1
            ans = str('All_Suitable')

        if answer == 2:
            overhang_angle_accept = overhang_angle_accept + 1
            overhang_length_non_accept = overhang_length_non_accept + 1
            ans = str('One OA_Suitable & One OL_Non_Suitable')

        if answer == 1:
            overhang_angle_non_accept = overhang_angle_non_accept + 1
            overhang_length_accept = overhang_length_accept + 1
            ans = str('One OA_Non_Suitable & One OL_Suitable')

        if answer == 0:
            overhang_angle_non_accept = overhang_angle_non_accept + 1
            overhang_length_non_accept = overhang_length_non_accept + 1
            ans = str('All_Non_Suitable')

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.voxels(comp_results[i])  # 29
        # plt.savefig('C:/Users/user/PycharmProjects/Feature_Recognition/완료/CaseDesign/no.8/' + str(i) + '.One Overhang Angle & One Overhang Length_' + str(ans) +'.png')
        # plt.close()

    if feature_index == 10:  # Two overhang & Bridge
        overhang_angle_count = overhang_angle_count + 2
        bridge_count = bridge_count + 1
        evaluation_y = two_overhang_bridge_evaluation_model.predict(answer_x)
        answer = np.argmax(evaluation_y[0])
        print("Two Overhang Angles & One Bridge Length")
        print(answer)

        if answer == 5:
            overhang_angle_accept = overhang_angle_accept + 2
            bridge_accept = bridge_accept + 1
            ans = str('All_Suitable')

        if answer == 4:
            overhang_angle_accept = overhang_angle_accept + 2
            bridge_non_accept = bridge_non_accept + 1
            ans = str('Two OA_Suitable & One BL_Non_Suitable')

        if answer == 3:
            overhang_angle_accept = overhang_angle_accept + 1
            overhang_angle_non_accept = overhang_angle_non_accept + 1
            bridge_accept = bridge_accept + 1
            ans = str('One OA_Suitable & One OA_Non_Suitable & One BL_Suitable')

        if answer == 2:
            overhang_angle_accept = overhang_angle_accept + 1
            overhang_angle_non_accept = overhang_angle_non_accept + 1
            bridge_non_accept = bridge_non_accept + 1
            ans = str('One OA_Suitable & One OA_Non_Suitable & One BL_Non_Suitable')

        if answer == 1:
            overhang_angle_non_accept = overhang_angle_non_accept + 2
            bridge_accept = bridge_accept + 1
            ans = str('Two OA_Non_Suitable & One BL_Suitable')

        if answer == 0:
            overhang_angle_non_accept = overhang_angle_non_accept + 2
            bridge_non_accept = bridge_non_accept + 1
            ans = str('All_Non_Suitable')

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.voxels(comp_results[i])  # 29
        # plt.savefig('C:/Users/user/PycharmProjects/Feature_Recognition/완료/CaseDesign/no.8/' + str(i) + '.Two Overhang Angles & One Bridge Length_' + str(ans) +'.png')
        # plt.close()

print("Hole diameters ----- The number of features:", hole_count, " Suitable:", hole_accept," Non-suitable:", hole_non_accept)
print("Overhang lengths ----- The number of features:", overhang_length_count, " Suitable:", overhang_length_accept," Non-suitable:", overhang_length_non_accept)
print("Overhang angles ----- The number of features:", overhang_angle_count, " Suitable:", overhang_angle_accept," Non-suitable:", overhang_angle_non_accept)
print("Overhang corners ----- The number of features:", overhang_corner_count, " Suitable:", overhang_corner_accept," Non-suitable:", overhang_corner_non_accept)
print("Overhang inners ----- The number of features:", overhang_inner_count, " Suitable:", overhang_inner_accept," Non-suitable:", overhang_inner_non_accept)
print("Bridge lengths ----- The number of features:", bridge_count, " Suitable:", bridge_accept," Non-suitable:", bridge_non_accept)
print("Pin diameters ----- The number of features:", pin_count, " Suitable:", pin_accept," Non-suitable:", pin_non_accept)
print("Wall thickness ----- The number of features:", wall_count, " Suitable:", wall_accept," Non-suitable:", wall_non_accept)
print("Computational time: ", time.time() - start_time)