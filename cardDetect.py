# Created on Dec 29th 2019
# Author: Changzhi Cai
# Contact me: caichangzhi97@gmail.com

# import package
from imutils import contours
import cv2
import numpy as np

# determine the credit card type
FIRST_NUMBER = {
	"3": "American Express",
	"4": "Visa",
	"5": "MasterCard",
	"6": "Discover Card"
}

def sort_contours(cnts,method = "left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
        
    # use a minimum restangular to enclose the target area
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts,boundingBoxes) = zip(*sorted(zip(cnts,boundingBoxes),
                                        key = lambda b:b[1][i],reverse = reverse))
    return cnts,boundingBoxes

def resize(image,width = None,height = None,inter = cv2.INTER_AREA):
    dim = None
    (h,w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height/float(h)
        dim = (int(w*r),height)
    else:
        r = width/float(w)
        dim = (width,int(h*r))
    resized = cv2.resize(image,dim,interpolation = inter)
    return resized

# show the image
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(500)
    cv2.destroyAllWindows()

# read a reference image
img = cv2.imread('ocr_a_reference.png')
cv_show('img',img)

# convert it to a grayscale image
ref = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv_show('ref',ref)

# convert it to a binary image
ref = cv2.threshold(ref,10,255,cv2.THRESH_BINARY_INV)[1]
cv_show('ref',ref)

# calculate the contour
# cv2.findContours() only accept binary image
# cv2.RETR_EXTERNAL only detect outer contour
# cv2.CHAIN_APPROX_SIMPLE only keep the destination location
# each element from the returned list is a contour of the image
ref_,refCnts,hierarchy = cv2.findContours(ref.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img,refCnts,-1,(0,0,255),3) 
cv_show('img',img)
print (np.array(refCnts).shape)

# sorting from left to right, from up to down
refCnts = sort_contours(refCnts,method="left-to-right")[0] 
digits = {}

# traversal every contour
for (i,c) in enumerate(refCnts):
    
    # calculate bounded rectangle and resize it
	(x,y,w,h) = cv2.boundingRect(c)
	roi = ref[y:y+h,x:x+w]
	roi = cv2.resize(roi,(57,88))
    
    # each number has a template
	digits[i] = roi

# initialize convoluton kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

# read the input image and do preprocesing
image = cv2.imread('credit_card_03.png')
cv_show('image',image)
image = resize(image,width = 300)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv_show('gray',gray)

# tophat operation and emphasize bright areas
tophat = cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,rectKernel)
cv_show('tophat',tophat)

# ksize = -1 represents 3*3 kernel
gradX = cv2.Sobel(tophat,ddepth = cv2.CV_32F,dx = 1,dy = 0,ksize = -1)

gradX = np.absolute(gradX)
(minVal,maxVal) = (np.min(gradX),np.max(gradX))
gradX = (255*((gradX-minVal)/(maxVal-minVal)))
gradX = gradX.astype("uint8")

print(np.array(gradX).shape)
cv_show('gradX',gradX)

# connect numbers together with close operation (expansion, then corrosion)
gradX = cv2.morphologyEx(gradX,cv2.MORPH_CLOSE,rectKernel) 
cv_show('gradX',gradX)

# THRESH_OTSU will automatically find a suitable threshold, suitable for double peaks. 
# the threshold parameter needs to be set to 0.
thresh = cv2.threshold(gradX,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
cv_show('thresh',thresh)

# do another close operation
thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,sqKernel)
cv_show('thresh',thresh)

# calculate the contour
thresh_,threshCnts,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img,cnts,-1,(0,0,255),3)
cv_show('img',cur_img)
locs = []

# traverse the contours
for (i,c) in enumerate(cnts):
    
    # calculate the rectangle
    (x,y,w,h) = cv2.boundingRect(c)
    ar = w/float(h)
    
    # according to the actual image, leave the suitable results
    if ar>2.5 and ar<4.0:
        if(w>40 and w<55) and (h>10 and h<20):
            locs.append((x,y,w,h))

# sort the suitable contours from left to right
locs = sorted(locs,key = lambda x:x[0])
output = []

# traverse numbers contained in each contour
for (i,(gX,gY,gW,gH)) in enumerate(locs):
    
    # initialize the list of group digits
    groupOutput = []
    
    # extract each group according to the location
    group = gray[gY-5:gY+gH+5,gX-5:gX+gW+5]
    cv_show('group',group)
    
    # preprocess
    group = cv2.threshold(group,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    cv_show('group',group)
    
    # calculate the contour of each group
    group_,digitCnts,hierarchy = cv2.findContours(group.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = contours.sort_contours(digitCnts,method = "left-to-right")[0]
    
    # calculate each value in each group
    for c in digitCnts:
        
        # find the outline of the current value, resize to the appropriate size
        (x,y,w,h) = cv2.boundingRect(c)
        roi = group[y:y+h,x:x+w]
        roi = cv2.resize(roi,(57,88))
        cv_show('roi',roi)
        
        # calculate match score
        scores = []
        
        # calculate each score in template
        for(digit,digitROI) in digits.items():
            
            # match template
            result = cv2.matchTemplate(roi,digitROI,cv2.TM_CCOEFF)
            (_,score,_,_) = cv2.minMaxLoc(result)
            scores.append(score)
        
        # find the most suitable number
        groupOutput.append(str(np.argmax(scores)))
    
    # draw it
    cv2.rectangle(image,(gX-5,gY-5),(gX+gW+5,gY+gH+5),(0,0,255),1)
    cv2.putText(image,"".join(groupOutput),(gX,gY-15),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),2)
    
    # get the final result
    output.extend(groupOutput)

# print the final result
print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
cv2.imshow("Image",image)
cv2.waitKey(0)