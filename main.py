import cv2
import numpy as np
from PIL import Image

# load the data we generated previously
samples = np.loadtxt('general-samples.data', np.float32)
responses = np.loadtxt('general-responses.data', np.float32)
responses = responses.reshape((responses.size,1))
 
# train the KNN model
model = cv2.KNearest()
model.train(samples, responses)
 

image= cv2.imread('captcha.jpg')
#cv2.imwrite("imagefinal.jpg",image)
out = np.zeros(image.shape, np.uint8)


gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

'''
height ,width, depth = image.shape

15,26 36 48 59 73 
minh=
minw=
maxh
maxw
'''
#cv2.imwrite("a.jpg",thresh)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    [x, y, w, h] = cv2.boundingRect(contour)
    print x,y,w,h
    
    if cv2.contourArea(contour) > 50:
       
        # contour is sufficiently large to possibly be a number
        if (  ) :
            
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # select the sample, resize to 10x10 and then vectorize
            roi = thresh[y:y+h, x:x+w]
            cv2.imwrite("temp.jpg",roi)
            roi= cv2.imread('temp.jpg')

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
#cv2.imwrite("result.jpg",image)
# show the results
cv2.imshow('im',image)
cv2.imshow('out',out)
cv2.waitKey(0)
