#!/usr/bin/env python 

import sys
import cv2
import numpy as np
from scipy.spatial import distance
from scipy.optimize import least_squares
import math as m
import time
from HelperFunctions_v4 import *

code_proc_time = time.time()

Data_Num = 7              # 1-7
Data_Num = Data_Num - 1
disparity_type = "BM"     # "BM+WLS","BM"
select_type = "Inlier"    # "Inlier","Outler"
flag_record = 0           # 1-Map only, 2-Full Video
nPts_opt = 10             # Number of points for optimization

flag_display = 1

# Approx number of points to be returned after optical flow
nPts_track = 40
dist_thr = [10,25]        # Disparity Distance Threshold

Data = [ "Data Set/2009_09_08_drive_0010",
         "Data Set/2010_03_17_drive_0046",
         "Data Set/2010_03_05_drive_0023",
         "Data Set/2010_03_09_drive_0081",
         "Data Set/2010_03_09_drive_0020",
         "Data Set/2010_03_04_drive_0033",
         "Data Set/2010_03_09_drive_0019"]
Data_dir = Data[Data_Num]
Data_Frames = [1424, 967, 370, 341, 546, 399, 372]
Pause = False

# Reading the calibration file and getting the Projection Matrix
Calib_file = open(Data_dir + "_calib.txt",'r')
Calib_file_lines = Calib_file.readlines()
P1_roi = np.zeros(12)
P2_roi = np.zeros(12)
for j in range(12):
    P1_roi[j] = Calib_file_lines[12].split()[j+1]
    P2_roi[j] = Calib_file_lines[14].split()[j+1]
P1_roi = P1_roi.reshape(3,4)
P2_roi = P2_roi.reshape(3,4)
# Setting the camera parameters
f = P1_roi[0,0]		
base = -P2_roi[0,3]/P2_roi[0,0]			
cx = P1_roi[0,2]				
cy = P1_roi[1,2]

# Read Ground Truth File and initialize Map
GT_file = open(Data_dir + '/insdata.txt','r')
GT_file_lines = GT_file.readlines()
StartGT = [0,0]
CurGT = [0,0]
StartGT[0] = float(GT_file_lines[0].split()[4])
StartGT[1] = float(GT_file_lines[0].split()[5])
Map = np.zeros((20,20,3), np.uint8)
centre = int(Map.shape[0]/2)
cv2.circle(Map,(centre,centre),5,color=(0,255,0), thickness = -1)

currID = 0      # Current Frame ID 
prevID = 0      # Previous Frame ID
# Image Variables : Left Frame, Right Frame, Disparity Map, Disparity Mask
curr = [0,0,0,0] 
bgr = [0,0,0,0]
prev = [0,0,0,0]

Position = [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0]]
# Getting the initial pose of the vehicle
# So that VO can be compared with Ground Truth
CurGT[0] = float(GT_file_lines[1].split()[4])
CurGT[1] = float(GT_file_lines[1].split()[5])
GT_x = CurGT[0] - StartGT[0]
GT_y = CurGT[1] - StartGT[1]
theta = m.atan2(-GT_x,-GT_y)
IniRot = [[m.cos(theta) , 0, m.sin(theta), 0],
          [0            , 1, 0           , 0],
          [-m.sin(theta), 0, m.cos(theta), 0],
          [0            , 0, 0           , 1]]
Position = np.dot(Position,IniRot)

# Parameter for Save Video and the final Map
SaveName = str(Data_Num+1)+"_"+str(disparity_type)\
           +"_"+str(nPts_opt)+"_"+str(select_type)+"_new"
if flag_record == 2:
   fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
   if Data_Num == 0:
      out = cv2.VideoWriter("Video_"+SaveName+".avi",\
                            fourcc, 30.0, (1448,440))
   elif Data_Num == 1:
      out = cv2.VideoWriter("Video_"+SaveName+".avi",\
                            fourcc, 30.0, (1438,415))
   elif Data_Num == 2:
      out = cv2.VideoWriter("Video_"+SaveName+".avi",\
                            fourcc, 30.0, (1434,421))
   elif Data_Num == 3:
      out = cv2.VideoWriter("Video_"+SaveName+".avi",\
                            fourcc, 30.0, (1426,418))
   elif Data_Num == 4:
      out = cv2.VideoWriter("Video_"+SaveName+".avi",\
                            fourcc, 30.0, (1426,418))
   elif Data_Num == 5:
      out = cv2.VideoWriter("Video_"+SaveName+".avi",\
                            fourcc, 30.0, (1432,421))
   elif Data_Num == 6:
      out = cv2.VideoWriter("Video_"+SaveName+".avi",\
                            fourcc, 30.0, (1426,418))
      
np.set_printoptions(suppress=True)

# Setting parameters based on disparity type selected
if disparity_type == "BM":
   left_matcher = cv2.StereoBM_create(\
       numDisparities=16*10, blockSize=15)
elif disparity_type == "BM+WLS":
   left_matcher = cv2.StereoBM_create(\
       numDisparities=16*10, blockSize=15)
   right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
   wls_filter = cv2.ximgproc.createDisparityWLSFilter(\
       matcher_left=left_matcher)
   wls_filter.setLambda(8000)
   wls_filter.setSigmaColor(0.8)
else:
   print ("Incorrect Disparity Method Selected")
   sys.exit()

i = 0
k = 0

if flag_display == 1:
    cv2.namedWindow("Stereo VO MAP")

print ("CODE STARTED")
# Starting the process
while(k!=27 and i<Data_Frames[Data_Num]):
   if Pause == False:
      currID = i

      # Reading LEft and Right Stereo Pair
      Left_Img_path = Data_dir + "/I1_" + str(i).zfill(6) + ".png"
      curr[0] = cv2.imread(Left_Img_path,0)

      Right_Img_path = Data_dir + "/I2_" + str(i).zfill(6) + ".png"
      curr[1] = cv2.imread(Right_Img_path,0)

      # Calculating Disparity
      if disparity_type == "BM":
         curr[2] = np.int16(left_matcher.compute(curr[0],curr[1]))
      elif disparity_type == "BM+WLS":
         displ = np.int16(left_matcher.compute(curr[0],curr[1]))
         dispr = np.int16(right_matcher.compute(curr[1],curr[0]))
         curr[2] = wls_filter.filter(displ, curr[0], None, dispr)

      # Normalizing Disparity Map for Visualization
      disp_view = cv2.normalize(curr[2], None, beta=0,\
                alpha=np.amax(curr[2])/16, norm_type=cv2.NORM_MINMAX);
      disp_view = np.uint8(disp_view)

      # Dynamic Disparity Mask Thresholding
      temp_thr = dist_thr[0]
      curr[3] = cv2.inRange(\
          disp_view,int(f*base/dist_thr[1]),int(f*base/dist_thr[0]))
      while np.sum(curr[3]) < 20000000:
          temp_thr -= 1
          curr[3] = cv2.inRange(\
              disp_view,int(f*base/dist_thr[1]),int(f*base/temp_thr))
          if temp_thr == 3:
              break

      # Trimming Bottom, Left, Right part in the mask because
      # Bottom : Mostly Road
      # Top : Mostly Sky
      # Right : Features will not be there in next frame
      rows = curr[3].shape[0]
      cols = curr[3].shape[1]
      curr[3][int(3*rows/4):,:] = 0
      curr[3][:int(1*rows/10),:] = 0
      curr[3][:,int(17*cols/20):] = 0

      # Reading Ground Truth for comparison
      CurGT[0] = float(GT_file_lines[currID].split()[4])
      CurGT[1] = float(GT_file_lines[currID].split()[5])

      bgr[0] = cv2.cvtColor(curr[0],cv2.COLOR_GRAY2BGR)
      # bgr[1] = cv2.cvtColor(curr[1],cv2.COLOR_GRAY2BGR)
      if currID == prevID + 1:
         # Finding fast features with Disparity Mask and Binning
         Pts_1 = find_keypoints(prev[0],prev[3],20,20,1)
         # Tracking the features detected
         Pts_1,Pts_2 = track_keypoints(\
             prev[0],curr[0],Pts_1,nPts_track)
         # Finding 3D Coordinates for features tracked
         Pts_1,Pts_2,Pts3D_1,Pts3D_2 = \
             Calc_3DPts(prev[2],Pts_1,curr[2],Pts_2,f,base,cx,cy,16)

         # Choosing method to select points for optimization
         if select_type == "Outlier":
             Pts_1F, Pts_2F, Pts3D_1F, Pts3D_2F = \
                 find_bestPts_OR(Pts_1,Pts_2,Pts3D_1,Pts3D_2,nPts_opt)
         elif select_type == "Inlier":
             clique = find_bestPts_ID(Pts3D_1,Pts3D_2,nPts_opt)
             Pts_1F = [Pts_1[i] for i in clique]
             Pts_2F = [Pts_2[i] for i in clique]
             Pts3D_1F = [Pts3D_1[i] for i in clique]
             Pts3D_2F = [Pts3D_2[i] for i in clique]
         else:
             print ("Incorrect Feature Selection Method")
             sys.exit()

         # Homogenizing the coordinates
         homo = np.ones((len(Pts3D_1F),1))
         Pts_1F = np.hstack((Pts_1F,homo))
         Pts_2F = np.hstack((Pts_2F,homo))
         Pts3D_1F = np.hstack((Pts3D_1F,homo))
         Pts3D_2F = np.hstack((Pts3D_2F,homo))

         # Running the optimization
         dSeed = np.zeros(len(Pts3D_1F))
         optRes = least_squares(mini, dSeed, method='lm', \
                  max_nfev=200,args=(Pts3D_1F, Pts3D_2F, Pts_1F,\
                  Pts_2F,P1_roi))

         # Finding Rotation and Translation
         Rmat = genEulerZXZMatrix(\
             optRes.x[0], optRes.x[1], optRes.x[2])
         Trans = np.array(\
             [[optRes.x[3]], [optRes.x[4]], [optRes.x[5]]])

         # Updating the odometry
         newPosition = np.vstack(\
             (np.hstack((Rmat,Trans)),[0, 0, 0, 1]))
         Position = np.dot(Position,newPosition)

         # Processing Ground Truth for plotting
         GT_x = CurGT[0] - StartGT[0]
         GT_y = CurGT[1] - StartGT[1]
        
         # Resizing map dynamically
         while (centre+int(GT_x) >= Map.shape[0]-25 or\
                centre-int(Position[0,3]) >= Map.shape[0]-25 or\
                centre-int(GT_y) >= Map.shape[1]-25 or \
                int(Position[2,3])+centre >= Map.shape[1]-25):
             Map = np.insert(Map,len(Map[0]),0,axis=1)
             Map = np.insert(Map,len(Map),0,axis=0)
         while (centre+int(GT_x) <= 25 or\
                centre-int(Position[0,3]) <= 25 or\
                centre-int(GT_y) <= 25 or\
                int(Position[2,3])+centre <= 25):
             Map = np.insert(Map,0,0,axis=1)
             Map = np.insert(Map,0,0,axis=0)
             centre+=1
         
         # Plotting Ground Truth point in RED
         cv2.circle(Map,(centre+int(GT_x),centre-int(GT_y)),\
                    2,color=(0,0,255),thickness = -1)
         # Plotting VO point in Green
         cv2.circle(Map, (centre-int(Position[0,3]),\
                    int(Position[2,3])+centre),2,\
                    color=(0,255,0), thickness = -1)

         # Drawing the tracked features
         for j in range(len(Pts_2)):
             point = Pts_2[j,:]
             cv2.circle(bgr[0], (int(point[0]), int(point[1])), 10,\
                        color=(0,0,255), thickness = 3)
         # Drawing the features for optimization
         for j in range(len(Pts_2F)):
             point = Pts_2F[j,:]
             cv2.circle(bgr[0], (int(point[0]), int(point[1])), 10,\
                        color=(0,255,0), thickness = 3)
         
      elif prevID != 0:
         print ("Frames missed... Stopping execution")
         break

      prev[0] = curr[0]
      prev[1] = curr[1]
      prev[2] = curr[2]
      prev[3] = curr[3]
      prevID = currID
      i = i + 1

   # Processing data for display purposes
   if flag_display == 1:
       bgr[2] = cv2.cvtColor(disp_view,cv2.COLOR_GRAY2BGR)
       bgr[3] = cv2.cvtColor(curr[3],cv2.COLOR_GRAY2BGR)

       bgr[2] = cv2.resize(bgr[2],None,fx=0.5, fy=0.5,\
                           interpolation = cv2.INTER_CUBIC)
       bgr[3] = cv2.resize(bgr[3],None,fx=0.5, fy=0.5,\
                           interpolation = cv2.INTER_CUBIC)

       display_image = np.concatenate((bgr[2], bgr[3]),1)
       if display_image.shape[1]>=bgr[0].shape[1]:
          display_image = np.concatenate(\
              (bgr[0],display_image[:,0:bgr[0].shape[1],:]),0)
       else:
          display_image = np.concatenate(\
              (bgr[0][:,0:display_image.shape[1],:], display_image),0)

       display_image = cv2.resize(display_image,None,\
                    fx=0.75, fy=0.75,interpolation = cv2.INTER_CUBIC)
       scale = float(display_image.shape[0])/float(Map.shape[0])
       display_map = cv2.resize(Map,None,fx=scale,fy=scale,\
                                interpolation = cv2.INTER_CUBIC)
       display = np.concatenate((display_image, display_map),1)
       cv2.imshow("Stereo VO MAP",display)

       if flag_record == 2 and Pause == False \
          and currID <= Data_Frames[Data_Num]-1:
          out.write(display)
          if currID == Data_Frames[Data_Num]-1:
             out.release()
             print("Recording closed******************")

       if flag_record >= 1 and currID == Data_Frames[Data_Num]-1:
           cv2.imwrite("Map_"+SaveName+".png",Map)

       k = cv2.waitKey(1)

       if k == ord('p'):
          Pause = not Pause
       elif k == 27:
          break
   else:
       if (i%100 == 1):
           print (100*i/Data_Frames[Data_Num], "% completed")

# Displaying the processing time
code_proc_time = time.time() - code_proc_time
print ("CODE COMPLETED")
print(i/code_proc_time, "FPS")
if i == Data_Frames[Data_Num]:
    cv2.waitKey(0)
cv2.destroyAllWindows()
