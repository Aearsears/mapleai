# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 20:44:23 2020

"""
import cv2 as cv
from PIL import Image
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import math
#import pytesseract
import itertools
MIN_MATCH_COUNT = 10

GREY_MIN = (110,-6,78)
GREY_MAX = (150,24,171)

#input images are 800x600
def main():
    print(getMainCharName())




def oneTemplate():
    img = cv.imread('path')
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img2 = img_gray.copy()
    template = cv.imread('path',0)
    w, h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    for meth in methods:
        img_gray = img2.copy()
        method = eval(meth)
        # Apply template Matching
        res = cv.matchTemplate(img_gray,template,method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv.rectangle(img_gray,top_left, bottom_right, 255, 2)
        plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img_gray,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
        plt.show()


def manyTemplates():
    img_rgb = cv.imread('path')
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    template = cv.imread('path',0)
    
    w, h = template.shape[::-1]
    res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
    threshold = 0.30
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        
    cv.imwrite('result.jpg',img_rgb)
    result= cv.imread("result.jpg")
    cv.imshow("result",result)
    cv.waitKey(0);
    cv.destroyAllWindows();


def bruteForce():
    img = cv.imread('path',0)
    template = cv.imread('path',0)
    
    orb = cv.ORB_create()
    
    kp1, des1 = orb.detectAndCompute(img,None)
    kp2, des2 = orb.detectAndCompute(template,None)
    
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
    matches = bf.match(des1,des2)
# Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
    img3 = cv.drawMatches(img,kp1,template,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()
    


def homogen():
    colourimage= cv.imread('path')
    img1 = cv.imread('path',0)          # queryImage
    img2 = cv.imread('path',0) # trainImage
    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
            
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w,d = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
    
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    img3 =  cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    plt.imshow(img3, 'gray'),plt.show()
    cv.imwrite("homo.jpg",img3)
    matchedpoints1=[]
    matchedpoints2=[]
    print(good[1].trainIdx)
   # print(kp1)
    #KP2 is the big one
    #KP1 is the small one
    print(kp2[good[1].trainIdx])
    
    for point in good:
        #small one
        matchedpoints1.append(kp1[point.queryIdx])
        #big image
        matchedpoints2.append(kp2[point.trainIdx])
    
   # for point in matchedpoints2:
       # print(point.pt)
        
    drawcircle = img2.copy()
    
    roundedtuple = (round(matchedpoints2[0].pt[0]),(round(matchedpoints2[0].pt[1])))
    print(matchedpoints2[0].pt)
    print(roundedtuple)
    #the third value is the radius of the circle
    drawcircle = cv.circle(img2,roundedtuple,10,(0,0,255),0)
    drawcircle = cv.rectangle(img2,(277,288),(297,308),(0,0,255),0)
    drawcircle=cv.cvtColor(drawcircle,cv.COLOR_GRAY2BGR)
    
    cv.imwrite("drawcircle.jpg",drawcircle)
    
    colourimage = cv.circle(colourimage,roundedtuple,10,(0,0,255),0)
    colourimage = cv.rectangle(colourimage,(277,288),(297,308),(0,0,255),0)
    
    cv.imwrite("colouredtargets.jpg",colourimage)
    #drawcircle=cv.cvtColor(drawcircle,cv.COLOR_GRAY2BGR)
                

def getMainCharName():
    
    config = ("-l eng --oem 1 --psm 7")
    path = '/path/to/file'
    img = cv.imread(path)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    threshold =180
    _, img_binarized = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)
    pil_img = Image.fromarray(img_binarized)
    print(pytesseract.image_to_string(pil_img,config=config))



def getEXP(image):
    #[ 32 254 222] [ 22 244 182] [ 42 264 262]
    exproi = image[595:596,15:799]
    #convert BGR2HSV
    hsv = cv.cvtColor(exproi,cv.COLOR_BGR2HSV)
    #ranges for the hp color
    # red_min = np.array([161,189,197], np.uint8)
    # red_max=np.array([181,209,277], np.uint8)
    yellow_min = (24,245,163)
    yellow_max = (46,265,243)
    #get the mask
    mask= cv.inRange(hsv,yellow_min,yellow_max)
    bluepixels = cv.countNonZero(mask)
    #print('redpixels:'+str(redpixels))
    mppercent= (bluepixels/(exproi.shape[0]*exproi.shape[1]))
    return round(mppercent*100,2)

def getMP(image):
    #TAKE THE GREY SPACE AND SUBTRACT THAT FROM THE REST OF THE PIXELS
# =============================================================================
#[122 121 129] [112 111  89] [132 131 169] =============================================================================
    #mp = cv.imread('HP/mpline.jpg')
    mproi = image[575:576,435:575]
    #convert BGR2HSV
    hsv = cv.cvtColor(mproi,cv.COLOR_BGR2HSV)
    #ranges for the hp color
    # red_min = np.array([161,189,197], np.uint8)
    # red_max=np.array([181,209,277], np.uint8)
    blue_min = (83,75,89)
    blue_max = (132,140,295)
    #get the mask
    mask= cv.inRange(hsv,GREY_MIN,GREY_MAX)
    bluepixels = cv.countNonZero(mask)
    #print('redpixels:'+str(redpixels))
    mppercent= (bluepixels/(mproi.shape[0]*mproi.shape[1]))
    return 100 - (int)(round(mppercent*100,0))

def show(image):
    cv.imshow('image',image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def getHP(image):
    #takes in a numpy array corresponding to an image, finds the HP region of interest and then returns the HP amount in percent
    #returns a percentage of health based on the pixels read in
    #from the 800x600 screenshot, the roi of interest is 
    #[559:560,437:573]
    # lower:[161 189 197] upper:[181 209 277]
    #read
    hproi = image[559:560,435:575]
    # cv.imshow('mask',hproi)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    #hp = cv.imread('HP/hpline.jpg')
    #convert BGR2HSV
    hsv = cv.cvtColor(hproi,cv.COLOR_BGR2HSV)
    #ranges for the hp color
    # red_min = np.array([161,189,197], np.uint8)
    # red_max=np.array([181,209,277], np.uint8)
    red_min = (155,116,175)
    red_max = (181,209,277)
    #get the mask
    mask= cv.inRange(hsv,GREY_MIN,GREY_MAX)
    redpixels = cv.countNonZero(mask)
    #print('redpixels:'+str(redpixels))
    hppercent= (redpixels/(hproi.shape[0]*hproi.shape[1]))
    return 100 - (int)(round(hppercent*100,0))
    # cv.imshow('mask',mask)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


def isRed(pixel):
    #red
    #print(pixel[0])
    if(205<=pixel[0]<=255):
        #green
        #print(pixel.item(1))
        if(37<=pixel[1]<=57 or 65<=pixel[1]<=144):
            #blue
            #print(pixel.item(2))
            if(97<=pixel[2]<=117 or 138<=pixel[2]<=175):
                return 1
    return 0

def isRedHue():
    
# =============================================================================
#     pixel, lower upper
#     [170 207 242] [160 197 202] [180 217 282]
#     [167 169 225] [157 159 185] [177 179 265]
#     [170 139 255] [160 129 215] [180 149 295]
#     [167 124 255] [157 114 215] [177 134 295]
# =============================================================================
    #hue, saturation, value
    red = np.uint8([[[0,0,255]]])
    hsv_red = cv.cvtColor(red,cv.COLOR_BGR2HSV)
    #For HSV, hue range is [0,179], saturation range is [0,255], and value range is [0,255]
    print(hsv_red)


if __name__ == "__main__":
    main()
