import os
import cv2 as cv

img = cv.imread('bird3.jpg')

#cv.imshow('Parrot',img)
print(img.shape)


#resize image 

rz_img=cv.resize(img, (320,400))
#cv.imshow('Parrot_2',rz_img)
print(rz_img.shape)

#cropping image 

crp_img = rz_img[80:390,80:300] #height width 
cv.imshow('Parrot_3',crp_img)

#colourspace
# IMG_GRAY=cv.cvtColor(crp_img, cv.COLOR_BGR2GRAY)
# IMG_RGB=cv.cvtColor(crp_img, cv.COLOR_BGR2RGB)
# IMG_HSV=cv.cvtColor(crp_img, cv.COLOR_BGR2HSV)

# cv.imshow('Parrot_4',IMG_GRAY)
# cv.imshow('Parrot_5',IMG_RGB)
# cv.imshow('Parrot_6',IMG_HSV)


#blur image

# blur_image=cv.blur(crp_img,(5,5))#kernel size K
# blur_gaussian=cv.GaussianBlur(crp_img,(5,5),5)
# blur_median=cv.medianBlur(crp_img,5)
# blur_blt=cv.bilateralFilter(crp_img,15,75,75) ## Apply bilateral filter with d = 15, sigmaColor = sigmaSpace = 5. 

# cv.imshow('blur',blur_image)
# cv.imshow('gblur',blur_gaussian)
# cv.imshow('mblur',blur_median)
# cv.imshow('bblblur',blur_blt)

#Threshold Image 

# gray_img=cv.cvtColor(crp_img,cv.COLOR_BGR2GRAY)
# ret, thresh= cv.threshold(gray_img,80,255,cv.THRESH_BINARY)


# adap= cv.adaptiveThreshold(gray_img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,9)
# rett, otsu = cv.threshold(gray_img,0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
# cv.imshow('bblblur',gray_img)
# cv.imshow('thresh',thresh)
# cv.imshow('Adpative',adap)
# cv.imshow('otsu',otsu)

#@@@ EDGE DETECTOR

# img_cedge=cv.Canny(crp_img,200,100)

# cv.imshow('cannyedge',img_cedge)

# #Drawing

# #line 
# line=cv.line(crp_img, (100,100),(150,100),(255,255,255),30)

# cv.imshow('line',line)


# #rectangle 
# rectangle=cv.rectangle(crp_img, (120,120),(150,150),(255,0,255),5)

# cv.imshow('rectangle',rectangle)


# #circle 
# crl=cv.circle(crp_img, (10,100),25,(0,255,255),7)

# cv.imshow('Circle',crl)


# #text 
# txt=cv.putText(crp_img,'Hello' ,(80,50),cv.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,155,255),)

# cv.imshow('txt',txt)
# cv.waitKey(0)

#Contours

gray=cv.cvtColor(crp_img,cv.COLOR_BGR2GRAY)
res, thresh=cv.threshold(gray,127,255,cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

for cont in contours:
    if cv.contourArea(cont)>2:
        x1,y1,w,h=cv.boundingRect(cont)
        cv.rectangle(crp_img,(x1,y1),(x1+w,y1+h),(0,255,0),2)
    #print(cv.contourArea(cont))  

cv.imshow('gray', gray)
cv.imshow('Contours', crp_img)

cv.waitKey(0)
