import cv2
import ImageEnhance #contrast

import numpy as np
from PIL import Image

# load the data we generated previously
samples = np.loadtxt('general-samples.data', np.float32)
responses = np.loadtxt('general-responses.data', np.float32)
responses = responses.reshape((responses.size,1))
 
# train the KNN model
model = cv2.KNearest()
model.train(samples, responses)



image = Image.open("captcha6.jpg")
nx, ny = image.size
image = image.resize((int(nx*5), int(ny*5)), Image.BICUBIC)
image.save("temp2.jpg")
enh = ImageEnhance.Contrast(image)
#enh.enhance(1.3).show("30% more contrast")

image= cv2.imread('temp2.jpg')
out = np.zeros(image.shape, np.uint8)



gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray,(5,5),0)
cv2.imwrite("a.jpg",blur)
ret,thresh = cv2.threshold(blur,127,255,cv2.THRESH_TOZERO)

#ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite("BlackWhite.jpg",thresh)
img = cv2.imread('BlackWhite.jpg')
im3=img.copy()


# convert to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imwrite("b.jpg",gray)
'''
# smooth the image to avoid noises
gray = cv2.medianBlur(gray,5)
'''



'''
# Apply adaptive threshold
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
cv2.imwrite("c.jpg",thresh_color)

# apply some dilation and erosion to join the gaps
thresh = cv2.dilate(thresh,None,iterations = 3)
thresh = cv2.erode(thresh,None,iterations = 2)
cv2.imwrite("d.jpg",thresh_color)
'''
# Find the contours

'''
az 420ta man 40 ta sho mikham mishe ?
az 110 ta 70 ta mikham>>>>>>

'''
height ,width, depth = image.shape
print height ,width

minValidHeight=height*0.3
minValidWidth=width*0.08
maxValidHeight=height*0.99
maxValidWidth=width*0.5
print  minValidWidth, minValidHeight ,maxValidWidth,maxValidHeight 



contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
a=0
for contour in contours:
    [x, y, w, h] = cv2.boundingRect(contour)
    print w,h
    a+=1
    lst=[]
    found =False
    

    if((w>minValidWidth or h>minValidHeight ) and h<maxValidHeight and w<maxValidWidth):
        cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,0),1)
        for l in lst:
            if(l[0]>x and l[2]<w and l[0]+l[2]<x+w and l[1]>y and l[3]<h and l[1]+l[3]<y+h):
                l[0]=x;
                l[1]=y
                l[2]=w;
                l[3]=h;
                found=True
                break
        if(found==False):
            lst.append([x,y,w,h])

        # select the sample, resize to 10x10 and then vectorize
        roi = thresh[y:y+h, x:x+w]
        cv2.imwrite("temp"+str(x)+".jpg",roi)




        roi= cv2.imread('temp'+str(x)+'.jpg')

        roi_small = cv2.resize(roi,(10,10))
        roi_small = roi_small.reshape((-1,100))
        roi_small = np.float32(roi_small)

        # find nearest neighbor
        retval, results, neigh_resp, dists = model.find_nearest(roi_small, k = 1)
        # extra parens?
        string = str(int((results[0][0])))

        # write the result the output image
        cv2.putText(out, string, (x, y+h), 0, 1, (0, 255, 0))
        print string

cv2.imshow('im',image)
cv2.imshow('out',out)
cv2.waitKey(0)
