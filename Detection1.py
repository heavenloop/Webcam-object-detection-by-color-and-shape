# -*- coding: utf-8 -*-
import numpy as np
import cv2


#make the first cacpture with the webcam
cap = cv2.VideoCapture(0)

#define the range values that you want to threshold (doesnt matter, when yiu run the program you can change them with the 4 buttom an 6 buttom)
maxthresh = 110
minthresh = 100        

#function to make image bigger or smaller
def resize (img):
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_dim= max(img.shape)
    scale = 800/img_dim# <------------- if you want your image bigger or smaller change the value
    img= cv2.resize(img,None, fx=scale, fy=scale)
    return img
#function for threshold
def process (img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,maxthresh,minthresh,1)#threshold
    return thresh 
# function to find contours wich is base for detect structure based on the shape
def find_biggest_contour(image):
    # Copy
    image = image.copy()
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#Lets start the video    
while (True):
    ver,img= cap.read()                 #Read every capture
    img= resize(img)                    #resize the image
    thresh= process(img)                #make threshold
#%%    
    #push 6 button to increase the maximun value of thershold
    #push 4 button to decrease the maximun value of thershold
    mas = cv2.waitKey(20) & 0xff        
    menos = cv2.waitKey(20) & 0xff
    if mas == ord('6'):#
        maxthresh = maxthresh + 1
    if menos == ord('4'):#
        maxthresh = maxthresh - 1
        
    #push 6 button to increase the minimum value of thershold
    #push 4 button to decrease the minimum value of thershold
    mas = cv2.waitKey(10) & 0xff        
    menos = cv2.waitKey(10) & 0xff
    if mas == ord('3'):#
        maxthresh = minthresh + 1
    if menos == ord('1'):#
        maxthresh = minthresh - 1        
#%%   
     #DETECTION BASED ON COLOR 
     
    image_blur = cv2.GaussianBlur(img, (7, 7), 0)
    #t unlike RGB, HSV separates luma, or the image intensity, from
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)
    # Filter by colour
    #minimum red amount, max red amount
    min_red = np.array([0, 100, 80])
    max_red = np.array([10, 256, 256])
    #layer
    mask1 = cv2.inRange(image_blur_hsv, min_red, max_red)
    #birghtness of a color is hue
    # 170-180 hue
    min_red2 = np.array([170, 100, 80])
    max_red2 = np.array([180, 256, 256])
    mask2 = cv2.inRange(image_blur_hsv, min_red2, max_red2)
    # Combine masks
    mask = mask1 + mask2

#    # Clean up
#    #we want to circle our strawberry so we'll circle it with an ellipse
#    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
#    #morph the image. closing operation Dilation followed by Erosion. 
#    #It is useful in closing small holes inside the foreground objects, 
#    #or small black points on the object.
#    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#    #erosion followed by dilation. It is useful in removing noise
#    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
    img = img.copy()
    #input, gives all the contours, contour approximation compresses horizontal, 
    #vertical, and diagonal segments and leaves only their end points. For example, 
    #an up-right rectangular contour is encoded with 4 points.
    #Optional output vector, containing information about the image topology. 
    #It has as many elements as the number of contours.
    #we dont need it
#    big_strawberry_contour, mask_strawberries = find_biggest_contour(mask_clean)
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    #calculates the weightes sum of two arrays. in our case image arrays
    #input, how much to weight each. 
    #optional depth value set to 0 no need
    img = cv2.addWeighted(rgb_mask, 0.5, img, 0.5, 0)
    cv2.imshow('BASED ON COLOR',img)
#%%
    #DETECTION BASED ON SHAPE
    
    contours,h = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)[-2:] #encuentra contornos
    cv2.imshow('img1',thresh)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, .075 * cv2.arcLength(cnt, True), True)
        if len(approx)==3:#triangle
            cv2.drawContours(img,[cnt],0,(200,0,0),-1)#blue
        elif len(approx)==4:#square
           cv2.drawContours(img,[cnt],0,(0,200,0),-1)#green
        elif len(approx)>=5:
            area = cv2.contourArea(cnt)
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            circleArea = radius * radius * np.pi
            #if circleArea == area:
            cv2.drawContours(img, [cnt], 0, (0, 0, 200), -1)#red
           # ellipse = cv2.fitEllipse(contour)
    #add it
    #cv2.ellipse(image_with_ellipse, ellipse, green, 2, cv2.CV_AA)
    print("Threshold high",maxthresh)
    print("Threshold low",minthresh)
    cv2.imshow('BASED ON SHAPE',img)
    k = cv2.waitKey(20) & 0xff
    if k == ord('q'):# <------------------------PUSH q IN THE IMAGE WINDOW TO FINISH THE PROGRAM
        break
    


cv2.waitKey(0)
cv2.destroyAllWindows()   
