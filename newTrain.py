import cv2
import numpy as np
from datetime import datetime
import os

from PIL import Image
import hashlib, os, math, time
import Image
#from PIL import Image
import ImageEnhance
from pytesser import *
from urllib import urlretrieve
import math
import random
import glob


#iconset = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

iconset = ['0','0','0','3','3','3','3','3','3','3','3','5','5','5','5','7','7',
'7','7','7','7','7','7','7','4','4','4','4','4','4','4','4','2','2','2','2','2','2','2','2','2','2']
iconset = [0,1,20,5,6,7,8,10,11,15,17,2,12,21,22,3,4,
13,14,16,18,19,23,9,24,25,26,27,28,29,30,31,32,
33,34,35,36,37,38,39,40,41]
samples=np.empty((0,100))
responses =[]

print len(iconset)
filen=len(iconset)
for f in range(filen):
  #  im = cv2.imread('./iconset/'+str(f)+'.gif')
    img =Image.open('./iconset/'+str(f)+'.gif')
    print ('./iconset/'+str(f)+'.gif')
    img.convert('RGB').save ('img.jpg','JPEG')
    img= cv2.imread('img.jpg')
    im3= img.copy()
   # print "copy"
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #cv2.imwrite("A.gif",gray)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
    roi =thresh[0:height ,0:width]
    roismall = cv2.resize(roi,(10,10))
    responses.append(iconset[f])
    sample = roismall.reshape((1,100))
    samples = np.append(samples,sample,0)
  #  print"append"


responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print "training complete"


np.savetxt('general-samples.data',samples)
np.savetxt('general-responses.data',responses)







def train():
    samples = np.loadtxt('general-samples.data',np.float32)
    responses = np.loadtxt('general-responses.data',np.float32)
    responses = responses.reshape((1,responses.size))
    print "D"
    model = cv2.KNearest()
    model.train(samples,responses)
    return model

