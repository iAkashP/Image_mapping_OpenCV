import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import pygame, sys
from pygame.locals import *
import pygame.camera

width = 640
height = 480

#initialise pygame
# pygame.init()
# pygame.camera.init()
# cam = pygame.camera.Camera("/dev/video0",(width,height))
# cam.start()

# #setup window
# windowSurfaceObj = pygame.display.set_mode((width,height),1,16)
# pygame.display.set_caption('Camera')
MIN_MATCH_COUNT = 3

img1 = cv2.imread('TrainImg/nouturn.jpg',0)          # queryImage
vidrec=cv2.VideoCapture(0)

while(1):
    ret,imgdata = vidrec.read() # trainImage
    # imgdata = cam.get_image()
    img2 = pygame.surfarray.array3d(imgdata)
    # img2.swapaxes(0,1)
    # cv2.imshow("Result", img2)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    bp=len(kp2)
    if (bp>1):

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.4*n.distance:
                good.append(m)

        if len(good)>=MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            # dst = cv2.perspectiveTransform(pts,M)
            #
            # img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
            print "Match Found"

        else:
            # print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
            matchesMask = None
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)

        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

        # plt.imshow(img3, 'gray'),plt.show()
        cv2.imshow("Result", img3)
        cv2.waitKey(1)
cam.stop()
cv2.destroyAllWindows()
