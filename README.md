# 3D-2D-Stereo-VO
This was done as a part of Computer Vision course.<br />

In this project we implement feature based Stereo Visual Odometry (VO). This algorithm works based on tracking features between Frame at Time ‘T’ and ‘T+1’. As, we are using a stereo camera, the 3D world coordinates of the features can be calculated. The coordinates of
these features are used to find the pose of Frame ‘T+1’ wr.t Frame ‘T’. The pose is incrementally calculated every time instant as combined to build the final odometry of the mobile robot.<br />

An improvised algorithm was also tested to improve the accuracy of the visual odometry. The Stereo VO algorithm is tested on the widely used KITTI dataset (project by Karlsruhe Institute of Technology and Toyota Technological Institute). Using semantic segmentation for improving Stereo VO has been discussed briefly in the project. The Stereo VO generated using this algorithm are also shown in comparison to the ground truth provided in the KITTI dataset.<br />

One of the data sets used is available in the Data Set folder. Other data sets used for this can be found at : <br />
http://www.cvlibs.net/datasets/karlsruhe_sequences/

A report is also there to explain the various methods and algorithms used.


