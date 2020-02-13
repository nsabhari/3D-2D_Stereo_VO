#!/usr/bin/env python 

import sys
import cv2
import numpy as np
from scipy.spatial import distance
from scipy.optimize import least_squares
from math import *
debug = 0

# Function to get rotation matrix
def genEulerZXZMatrix(psi, theta, sigma):
    c1 = cos(psi)
    s1 = sin(psi)
    c2 = cos(theta)
    s2 = sin(theta)
    c3 = cos(sigma)
    s3 = sin(sigma)

    mat = np.zeros((3,3))

    mat[0,0] = (c1 * c3) - (s1 * c2 * s3)
    mat[0,1] = (-c1 * s3) - (s1 * c2 * c3)
    mat[0,2] = (s1 * s2)
    mat[1,0] = (s1 * c3) + (c1 * c2 * s3)
    mat[1,1] = (-s1 * s3) + (c1 * c2 * c3)
    mat[1,2] = (-c1 * s2)
    mat[2,0] = (s2 * s3)
    mat[2,1] = (s2 * c3)
    mat[2,2] = c2

    return mat

# Function for least squares optimization
def mini(dof,random_3d_1, random_3d_2, random_2d_1, random_2d_2,P):
    Rmat = genEulerZXZMatrix(dof[0], dof[1], dof[2])
    translationArray = np.array([[dof[3]], [dof[4]], [dof[5]]])
    temp = np.hstack((Rmat, translationArray))
    perspectiveProj = np.vstack((temp, [0, 0, 0, 1]))
    forward = np.matmul(P,perspectiveProj)
    backward = np.matmul(P, np.linalg.inv(perspectiveProj))
    numPoints = len(random_2d_1)
    errorA = np.zeros((numPoints,3))
    errorB = np.zeros((numPoints,3))
    pred2d_1 =[]
    pred2d_2 =[]
    for i in range(len(random_3d_1)):
    	pred2d_1.append(np.matmul(forward,random_3d_2[i]))
    	pred2d_1[i] = pred2d_1[i]/pred2d_1[i][-1]
    	pred2d_2.append(np.matmul(backward,random_3d_1[i]))
    	pred2d_2[i] = pred2d_2[i]/pred2d_2[i][-1]
    	error_1 = random_2d_1[i]-pred2d_1[i]
    	error_2 = random_2d_2[i]-pred2d_2[i]
    	errorA[i,:] = error_1.reshape(1,3)[0]
    	errorB[i,:] = error_2.reshape(1,3)[0]
    residual = np.vstack((errorA,errorB))
    return residual.flatten()

# Function to find fast features in the image
# Disparity Mask + Feature Binning
def find_keypoints(Image,Mask,Tile_H,Tile_W,nFeatures):
    featureEngine = cv2.FastFeatureDetector_create()
    H,W = Image.shape
    kp = []
    for y in range(0, H, Tile_H):
        for x in range(0, W, Tile_W):
            Patch_Img = Image[y:y+Tile_H, x:x+Tile_W]
            Patch_Mask = Mask[y:y+Tile_H, x:x+Tile_W]
            keypoints = featureEngine.detect(\
                Patch_Img,mask=Patch_Mask)
            for pt in keypoints:
                pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

            if (len(keypoints) > nFeatures):
                keypoints = sorted(keypoints,\
                                   key=lambda x: -x.response)
                for kpt in keypoints[0:nFeatures]:
                    kp.append(kpt)
            else:
                for kpt in keypoints:
                    kp.append(kpt)
        
    trackPts = cv2.KeyPoint_convert(kp)
    trackPts = np.expand_dims(trackPts, axis=1)

    global debug
    if debug == 1:
        print ("# Points Tracked : " + str(len(trackPts)))
    return trackPts

# Function to track keypoints from T to T+1 using LK Optical Flow
def track_keypoints(Image_1,Image_2,Pts_1,nPts):
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),maxLevel = 3,\
                      criteria = (cv2.TERM_CRITERIA_EPS | \
                      cv2.TERM_CRITERIA_COUNT, 50, 0.03))
    Pts_2, st, err = cv2.calcOpticalFlowPyrLK(\
                     Image_1,Image_2, Pts_1, None,\
                     flags=cv2.MOTION_AFFINE, **lk_params)

    # separate points that were tracked successfully
    ptTrackable = np.where(st == 1, 1,0).astype(bool)
    TrkPts_1 = Pts_1[ptTrackable, ...]
    TrkPts_2 = Pts_2[ptTrackable, ...]
    TrkPts_2 = np.around(TrkPts_2)
    global debug
    if debug == 1:
        print ("Points successfully tracked: " + str(len(Pts_2)))

    error = 4
    errTrackablePts = err[ptTrackable, ...]
    errThreshPts = np.where(errTrackablePts < \
                            error, 1, 0).astype(bool)
    # Dynamically change threshold to get required points
    while np.count_nonzero(errThreshPts) > nPts:
        error = round(error - 0.1,1)
        errThreshPts = np.where(errTrackablePts < \
                                error, 1, 0).astype(bool)

    while np.count_nonzero(errThreshPts) < nPts :
        error = round(error + 0.1,1)
        errThreshPts = np.where(errTrackablePts < \
                                error, 1, 0).astype(bool)
        if error >= 8:
            if debug == 1:
                print ("Max Limit Reached... Exiting loop")
            break

    TrkPts_1 = TrkPts_1[errThreshPts, ...]
    TrkPts_2 = TrkPts_2[errThreshPts, ...]
    if debug == 1:
        print ("Points with error less than " \
               + str(error) + " : " + str(len(TrkPts_1)))

    return TrkPts_1,TrkPts_2

# Finding 3D coordinate of selected features using Disparity Map
def Calc_3DPts(DisparityA,PointsA,DisparityB,PointsB,\
               f,base,cx,cy,scale):
    Pts_3DA = []
    Pts_3DB = []
    Pts_2DA = []
    Pts_2DB = []
    for j in range(len(PointsA)):
        PtA = PointsA[j]
        PtB = PointsB[j]
    
        dA = DisparityA[int(PtA[1])][int(PtA[0])]/scale
        dB = DisparityB[int(PtB[1])][int(PtB[0])]/scale
        if dA > 0 and dB > 0:
            Pts_3DA.append([base*(PtA[0] - cx)/dA,\
                            base*(PtA[1] - cy)/dA, f*base/dA])
            Pts_3DB.append([base*(PtB[0] - cx)/dB,\
                            base*(PtB[1] - cy)/dB, f*base/dB])
            Pts_2DA.append(PtA)
            Pts_2DB.append(PtB)
    return np.asarray(Pts_2DA),np.asarray(Pts_2DB),\
           np.asarray(Pts_3DA),np.asarray(Pts_3DB)

# Finding best points using Improved Inlier Detection
def find_bestPts_ID(point_cloud1,point_cloud2,minReq) :
    dist_difference = 0.05
    max_node = -1
    max_count = 0
    point_cloud1 = np.asarray(point_cloud1)
    point_cloud2 = np.asarray(point_cloud2)
    
    num_points = point_cloud1.shape[0]
    W = np.zeros((num_points,num_points))
    count = 0
    point_clouds_relative_dist = np.zeros((num_points,num_points))
    while max_node == -1:
        for i in range(num_points) : 
            diff_nodes_t1 = point_cloud1 - point_cloud1[i,:]
            diff_nodes_t2 = point_cloud2 - point_cloud2[i,:]
            dist_nodes_t1 = np.linalg.norm(diff_nodes_t1,axis=1)
            dist_nodes_t2 = np.linalg.norm(diff_nodes_t2,axis=1)
            abs_dist = abs(dist_nodes_t1 - dist_nodes_t2)

            point_clouds_relative_dist[i] = \
                np.asarray(abs_dist).T 
            wIdx = np.where(abs_dist < dist_difference)
            W[i,wIdx[0]] = 1
            count = np.sum(W[i,:])
            if count > max_count: 
                max_count = count
                max_node = i
        if max_count < minReq and dist_difference < 0.5 :
            max_count = 0
            max_node = -1
        if max_node == -1:
            dist_difference += 0.01
    count = 0
    clique = [max_node]

    while True :
        max_count = 0
        max_node = 0
        potentialnodes = list()
        Wsub = W[clique,:]
	# print(Wsub)
        for i in range(num_points) : 
            sumclique = np.sum(Wsub[:,i])
            if sumclique == len(clique) : 
                isin = True
            else : 
                isin = False
            if isin == True and i not in clique : 
                potentialnodes.append(i)
        max_count = 0
        max_node = 0 
        for i in range(len(potentialnodes)) : 
            Wsub = W[potentialnodes[i],potentialnodes]
            sumclique = np.sum(Wsub)
            if sumclique > max_count : 
                max_count = sumclique
                max_node = potentialnodes[i]

        if max_count == 0 :
            if len(clique) >= minReq : 
                break
            else :
                dist_difference += 0.05
                for k in range(num_points) : 
                    diff_nodes_t1 = point_cloud1 \
                                    - point_cloud1[k,:]
                    diff_nodes_t2 = point_cloud2 \
                                    - point_cloud2[k,:]
                    dist_nodes_t1 = \
                        np.linalg.norm(diff_nodes_t1,axis=1)
                    dist_nodes_t2 = \
                        np.linalg.norm(diff_nodes_t2,axis=1)
                    abs_dist = abs(dist_nodes_t1 - dist_nodes_t2)
                    point_clouds_relative_dist[k] = \
                        np.asarray(abs_dist).T 
                    wIdx = np.where(abs_dist < dist_difference)
                    W[k,wIdx[0]] = 1

        if len(clique) >= minReq or dist_difference > 10  :
            break
        clique.append(max_node)
    return clique

# Finding Best Points using Outlier Rejection
def find_bestPts_OR(Pts_1,Pts_2,Pts3D_1,Pts3D_2,minReq):
    if len(Pts3D_1) < 6:
        print("ERROR : Less than 6 points")
        print(len(Pts3D_1))
        sys.exit()
    Compare3D = np.zeros((len(Pts3D_1),len(Pts3D_1)))
    for i in range(len(Pts3D_1)):
        for j in range(len(Pts3D_1)):
            Dis_1 = distance.euclidean(Pts3D_1[i],Pts3D_1[j])
            Dis_2 = distance.euclidean(Pts3D_2[i],Pts3D_2[j])
            Compare3D[i,j] = abs(Dis_1-Dis_2)

    Sum3D = np.sum(Compare3D,axis = 1)
    FinalIndex = np.argsort(Sum3D)
    while len(Sum3D) > minReq:
        Compare3D = np.delete(Compare3D,FinalIndex[len(Sum3D)-1],0)
        Compare3D = np.delete(Compare3D,FinalIndex[len(Sum3D)-1],1)
        Pts_1 = np.delete(Pts_1,FinalIndex[len(Sum3D)-1],0)
        Pts_2 = np.delete(Pts_2,FinalIndex[len(Sum3D)-1],0)
        Pts3D_1 = np.delete(Pts3D_1,FinalIndex[len(Sum3D)-1],0)
        Pts3D_2 = np.delete(Pts3D_2,FinalIndex[len(Sum3D)-1],0)
        Sum3D = np.sum(Compare3D,axis = 1)
        FinalIndex = np.argsort(Sum3D)

    return np.asarray(Pts_1),np.asarray(Pts_2),\
           np.asarray(Pts3D_1),np.asarray(Pts3D_2)
