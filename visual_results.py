import numpy as np
import os
import torch
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import pptk

txt_path='/home/mark/下载/pclouds'
results_file='/home/mark/下载/pclouds/unoriented_our/eval'
point_cloud_file=txt_path
shape_list_filename='testset_all.txt'

def cos_angle(v1,v2):
    v1_norm = np.linalg.norm(v1, ord=2, axis=1, keepdims=True)
    v2_norm = np.linalg.norm(v2, ord=2, axis=1, keepdims=True)

    # 正则化
    v1_results = np.divide(v1, np.tile(v1_norm, [1, 3])+0.000001)
    v2_results = np.divide(v2, np.tile(v2_norm, [1, 3])+0.000001)

    m=np.sum(np.multiply(v1_results,v2_results),axis=1)
    m[m>1]=1
    m[m<-1]=-1
    return m
def l2_dist(v1,v2):
    v1_norm = np.linalg.norm(v1, ord=2, axis=1, keepdims=True)
    v2_norm = np.linalg.norm(v2, ord=2, axis=1, keepdims=True)

    # 正则化
    v1_results = np.divide(v1, np.tile(v1_norm, [1, 3])+0.000001)
    v2_results = np.divide(v2, np.tile(v2_norm, [1, 3])+0.000001)
    d=np.sum(np.square(v1_results - v2_results),axis=1)
    return d
shape_names = []
with open(os.path.join(txt_path, shape_list_filename)) as f:
   shape_names = f.readlines()
   shape_names = [x.strip() for x in shape_names]
   shape_names = list(filter(None, shape_names))

rms_all=[]
abs_rms_all=[]
acc_all=[]
pg10_all=[]
abs_pg1_all=[]
abs_pg3_all=[]
abs_pg5_all=[]
abs_pg10_all=[]
abs_pg15_all=[]
abs_pg20_all=[]
abs_pg25_all=[]
l2_all=[]
cosine_dist_all=[]
for i in range(len(shape_names)):#len(shape_names)
    print('Processing ' + shape_names[i] + '...')
    shape_name=shape_names[i]
    point_filename=os.path.join(point_cloud_file, shape_name + '.normals')
    point_cloud=np.loadtxt(point_filename).astype('float32')
    #ids_filename=os.path.join(results_file, shape_name + '.idx')
    #try:
    #    idx=np.loadtxt(ids_filename).astype('int')
    #except:
    #    print('avg_acc:', np.mean(acc_all))
    #    print('avg_rms:', np.mean(rms_all))
    #    print('avg_unoriented_rms:', np.mean(abs_rms_all))
    #    print('avg_l2:', np.mean(l2_all))
    #    print('avg_pg10:', np.mean(pg10_all))
     #   break

    #idx=torch.from_numpy(idx)
    results_filename=os.path.join(results_file, shape_name + '.normals')
    #normals_truth =point_cloud[idx, :]
    normals_truth =point_cloud
    normals_pred = np.loadtxt(results_filename).astype('float32')
    cosine_dist=cos_angle(normals_pred, normals_truth)
    angle_dif = np.rad2deg(np.arccos(cos_angle(normals_pred, normals_truth)))
    abs_angle_diff = np.rad2deg(np.arccos(abs(cos_angle(normals_pred, normals_truth))))
    angle_right = angle_dif < 90
    pg10 = angle_dif < 10
    abs_pg1 = abs_angle_diff < 1
    abs_pg3 = abs_angle_diff < 3
    abs_pg5 = abs_angle_diff < 5
    abs_pg10 = abs_angle_diff < 10
    abs_pg15 = abs_angle_diff < 15
    abs_pg20 = abs_angle_diff < 20
    abs_pg25 = abs_angle_diff < 25
    acc=float(np.sum(angle_right)/len(angle_right))
    pg10_acc=float(np.sum(pg10)/len(pg10))
    abs_pg1_acc=float(np.sum(abs_pg1)/len(abs_pg1))
    abs_pg3_acc=float(np.sum(abs_pg3)/len(abs_pg3))
    abs_pg5_acc=float(np.sum(abs_pg5)/len(abs_pg5))
    abs_pg10_acc=float(np.sum(abs_pg10)/len(abs_pg10))
    abs_pg15_acc=float(np.sum(abs_pg15)/len(abs_pg15))
    abs_pg20_acc=float(np.sum(abs_pg20)/len(abs_pg20))
    abs_pg25_acc=float(np.sum(abs_pg25)/len(abs_pg25))
    l2_distance=np.sqrt(np.mean(l2_dist(normals_pred, normals_truth))/3)
    print(shape_name,'acc:',acc)
    print(shape_name, 'pg10_acc:', pg10_acc)
    print(shape_name, 'rms:', np.sqrt(np.mean(np.square(angle_dif))))
    print(shape_name, 'unoriented_rms:', np.sqrt(np.mean(np.square(abs_angle_diff))))
    print(shape_name, 'l2_rms:', l2_distance)
    rms_all.append(np.sqrt(np.mean(np.square(angle_dif))))
    abs_rms_all.append(np.sqrt(np.mean(np.square(abs_angle_diff))))
    acc_all.append(acc)
    pg10_all.append(pg10_acc)
    abs_pg1_all.append(abs_pg1_acc)
    abs_pg3_all.append(abs_pg3_acc)
    abs_pg5_all.append(abs_pg5_acc)
    abs_pg10_all.append(abs_pg10_acc)
    abs_pg15_all.append(abs_pg15_acc)
    abs_pg20_all.append(abs_pg20_acc)
    abs_pg25_all.append(abs_pg25_acc)
    l2_all.append(l2_distance)
    #cosine_dist_all.append(1-np.mean(cosine_dist))
print('avg_acc:',np.mean(acc_all))
print('avg_rms:',np.mean(rms_all))
print('avg_unoriented_rms:',np.mean(abs_rms_all))
print('avg_l2:',np.mean(l2_all))
print('avg_pg10:',np.mean(pg10_all))
print('avg_abs_pg1:',np.mean(abs_pg1_all))
print('avg_abs_pg3:',np.mean(abs_pg3_all))
print('avg_abs_pg5:',np.mean(abs_pg5_all))
print('avg_abs_pg10:',np.mean(abs_pg10_all))
print('avg_abs_pg15:',np.mean(abs_pg15_all))
print('avg_abs_pg20:',np.mean(abs_pg20_all))
print('avg_abs_pg25:',np.mean(abs_pg25_all))
print('cosine_dist:',np.mean(cosine_dist_all))






