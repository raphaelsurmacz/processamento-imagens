import pandas as pd
import numpy as np
import numpy.random as rnd
import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

from os import listdir
from os.path import join
from sklearn.metrics import silhouette_score
import os

import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)

'''k-Means implementation'''
def distance(data, centroids, kind):
  #kind - euclidian

  k = len(centroids)
  cols=list()
  for i in range(1, (k + 1)):
    if kind=='euclidean':
      data[f'C{i}'] = ((centroids[i-1][0]-data.R)**2+(centroids[i-1][1]-data.G)**2+(centroids[i-1][2]-data.B)**2)**0.5

    cols.append(f'C{i}')
  data['Class'] = data[cols].abs().idxmin(axis=1)
  return data

def kmeans(data, K, kind):
  #print(10*'-', f'k={K}\tDistance={kind}', '-'*10)
  L = list()
  new_centroids = data.sample(K).values

  data = distance(data.copy(), new_centroids, kind)
  old_centroids = new_centroids.copy()
  new_centroids = np.array([data[data.Class == Class][['R', 'G', 'B']].mean().values for Class in data.loc[:,'C1':f'C{K}'].columns])
  i = 1
  #print(f'Iteration: {i}\tDistance: {abs(new_centroids.mean()-old_centroids.mean())}')
  while abs(new_centroids.mean()-old_centroids.mean())>0.001:
    L.append(abs(new_centroids.mean()-old_centroids.mean()))
    data = distance(data, new_centroids, kind)
    old_centroids = new_centroids.copy()
    new_centroids = np.array([data[data.Class == Class][['R', 'G', 'B']].mean().values for Class in data.loc[:,'C1':f'C{K}'].columns])
    #if np.isnan(new_centroids).any(): #in case there is an empty cluster
    i+=1
    #print(f'Iteration: {i}\tDistance: {abs(new_centroids.mean()-old_centroids.mean())}')
  #print(f"k-Means has ended with {i} iteratinons")
  return data, L

folder = os.getcwd()
imgInfo = []

def process_image(img, imgHSV):
    df = pd.DataFrame({'R': img[:,:,0].flatten(), 'G': img[:,:,1].flatten(), 'B': img[:,:,2].flatten()})
    df2 = pd.DataFrame({'R': imgHSV[:,:,0].flatten(), 'G': imgHSV[:,:,1].flatten(), 'B': imgHSV[:,:,2].flatten()})

    segmented_1, segmented_2, distances_1, distances_2 = {}, {}, {}, {}
    scores_1, scores_2 = {}, {}

    for k_value in [2, 3]:
        segmented_1[k_value], distances_1[k_value] = kmeans(df, k_value, 'euclidean')
        segmented_2[k_value], distances_2[k_value] = kmeans(df2, k_value, 'euclidean')
    
    for k_value in [2]:
      scores_1[k_value] = round(silhouette_score(segmented_1[k_value].loc[:, :'C2'], segmented_1[k_value].Class, metric='euclidean'), 2)
      scores_2[k_value] = round(silhouette_score(segmented_2[k_value].loc[:, :'C2'], segmented_2[k_value].Class, metric='euclidean'), 2)
    
    for k_value in [3]:
      scores_1[k_value] = round(silhouette_score(segmented_1[k_value].loc[:, :'C3'], segmented_1[k_value].Class, metric='euclidean'), 2)
      scores_2[k_value] = round(silhouette_score(segmented_2[k_value].loc[:, :'C3'], segmented_2[k_value].Class, metric='euclidean'), 2)
    
    return scores_1, scores_2

scores_rgb = {2: [], 3: []}
scores_hsv = {2: [], 3: []}

scores_rgb = {2: [], 3: []}
scores_hsv = {2: [], 3: []}

for filename in os.listdir(folder):
    if filename.endswith('.tif'):
        img = cv2.imread(os.path.join(folder, filename))
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        scores_rgb, scores_hsv = process_image(img, imgHSV)

        imgInfo.append((filename, scores_rgb, scores_hsv))

        # Testar borramento e recalcular scores
        for blur_size in [11, 13]:
            blurred_img = cv2.blur(img, (blur_size, blur_size))
            blurred_imgHSV = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)
            
            scores_rgb_blur, scores_hsv_blur = process_image(blurred_img, blurred_imgHSV)
            
            imgInfo.append((filename, blur_size, 'RGB', scores_rgb_blur))
            imgInfo.append((filename, blur_size, 'HSV', scores_hsv_blur))

# Cálculo das médias
mean_scores_rgb = {2: [], 3: []}
mean_scores_hsv = {2: [], 3: []}

for info in imgInfo:
    if info[1] == 'RGB':
        if len(info[2]) >= 2:  # Verificar se há dados para ambos os k_values
            mean_scores_rgb[2].append(info[2][2])
            mean_scores_rgb[3].append(info[2][3])
    elif info[1] == 'HSV':
        if len(info[2]) >= 2:  # Verificar se há dados para ambos os k_values
            mean_scores_hsv[2].append(info[2][2])
            mean_scores_hsv[3].append(info[2][3])

mean_blur_scoresRGB = {k: np.mean(scores) for k, scores in scores_rgb.items()}
mean_blur_scoresHSV = {k: np.mean(scores) for k, scores in scores_hsv.items()}

best_rgb_k = max(mean_blur_scoresRGB, key=mean_blur_scoresRGB.get)
best_rgb_score = mean_blur_scoresRGB[best_rgb_k]

best_hsv_k = max(mean_blur_scoresHSV, key=mean_blur_scoresHSV.get)
best_hsv_score = mean_blur_scoresHSV[best_hsv_k]

print(f"Melhor média de Scores RGB: k = {best_rgb_k}, score = {best_rgb_score}")
print(f"Melhor média de Scores HSV: k = {best_hsv_k}, score = {best_hsv_score}")

#print(imgInfo)

