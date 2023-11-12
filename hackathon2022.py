import math
import sys

import cv2
import cvzone
import numpy as np
import imutils
from matplotlib import pyplot as plt

sym_error_margin = 6
if len(sys.argv) > 1:
    sym_error_margin = int(sys.argv[1])
sym_precision = 1
if len(sys.argv) > 2:
    sym_precision = int(sys.argv[2])

np.set_printoptions(threshold=sys.maxsize, linewidth=9999)

#UTILITATE CONVERT BGR LA GRAYSCALE
def convertBGR2GS(source_img):
    img_gs = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    return img_gs

### CITIRE IMAGINE SI PRELUCRARE ###
imagine = cv2.imread('source.png')
dimensiuni = imagine.shape
imagineCOPY = imagine.copy()
imagineGS = cv2.cvtColor(imagineCOPY, cv2.COLOR_BGR2GRAY)
imagineGSBlur = cv2.GaussianBlur(imagineGS, (7, 7), 0)
(T, thresholded) = cv2.threshold(imagineGSBlur, 230, 255,cv2.THRESH_BINARY_INV)
thresholdedv2 = thresholded.copy()
### CITIRE IMAGINE SI PRELUCRARE ###

### DEBUG ###

### DEBUG ###

### CULOARE BG ###
bg = imagineCOPY
(channel_b, channel_g, channel_r) = cv2.split(bg)
channel_b = channel_b.flatten()
channel_g = channel_g.flatten()
channel_r = channel_r.flatten()
countsb = np.bincount(channel_b)
countsg = np.bincount(channel_g)
countsr = np.bincount(channel_r)
bgbgr=np.argmax(countsb),np.argmax(countsg),np.argmax(countsr)
### CULOARE BG ###

contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#PRELUCRARE MASCA THRESHHOLD#
cv2.fillPoly(thresholdedv2, pts=contours, color=255)
#PRELUCRARE MASCA THRESHHOLD#

ariiSortate = sorted(contours, key=cv2.contourArea,reverse=True)
perimetreSortate = sorted(contours, key=lambda c:cv2.arcLength(c, False))

ROI=[] # VECTOR OBIECTE CROP
ROIMASK=[] # VECTOR BINARIZARE CROP
nr=0
i=0

task2 = np.zeros((dimensiuni[0],dimensiuni[1], 3), np.uint8)
task2[:]=bgbgr

numarObiecte = len(contours)

for i in range(numarObiecte):
    x,y,w,h = cv2.boundingRect(contours[i])
    b, g, r = cv2.split(imagineCOPY[y:y+h,x:x+w])
    rgba = [b, g, r, thresholdedv2[y:y+h,x:x+w]]

    
    ROIMASK.append(thresholdedv2[y:y+h,x:x+w])
    ROIPNG = cv2.merge(rgba, 4)
    ROI.append(ROIPNG)
    
    
    cv2.imwrite("ROI_{}.png".format(nr), ROI[i]) # FIXME REMOVE / only for debugging
    if((ariiSortate[0] is contours[i]) or (perimetreSortate[0] is contours[i])):
        task2 = cvzone.overlayPNG(task2, ROI[i],[x,y])
    nr += 1


# VECTOR OBIECTE CROP @ GRAYSCALE
ROIGS = []

for i in range(len(ROI)):
    ROIGS.append(convertBGR2GS(ROI[i]))

# VECTOR LUMINOZITATE OBIECTE
ROIBR = []
ROIBRMAX = []
ROIBRMIN = []
max_br = None
min_br = None
for k in range(len(ROI)):
    sum = 0,
    for i in range(len(ROI[k])):
        for j in range(len(ROI[k][i])):
            if ROIMASK[k][i][j] == 255:
                sum += ROIGS[k][i][j]
    brightness = sum / ( (len(ROI[k]+1) * (len(ROI[k][1]+1))))
    ROIBR.append(brightness)
    if max_br is None or brightness > max_br: 
        max_br = brightness
        ROIBRMAX = ROI[k]
    if min_br is None or brightness < min_br: 
        min_br = brightness
        ROIBRMIN = ROI[k]

# CANVAS TASK 3 #
task3 = np.zeros((dimensiuni[0],dimensiuni[1], 3), np.uint8)
task3[:]=bgbgr
# CANVAS TASK 3 #

# AFISARE TASK 3 #
for i in range(len(ROI)):
    x,y,w,h = cv2.boundingRect(contours[i])
    if(ROIBRMAX is ROI[i] or ROIBRMIN is ROI[i]):
        task3 = cvzone.overlayPNG(task3, ROI[i],[x,y])
    nr += 1

# cv2.imshow("ROI", ROIMASK[11])

def get_container_canvas(source_img):
    h, w = source_img.shape
    bigger_size = round(math.sqrt( (h ** 2) + (w ** 2))) + 1

    container_canvas = np.zeros((bigger_size, bigger_size), np.uint8)

    return container_canvas

def get_rotated_roi(source_img, angle): 
    canvas = get_container_canvas(source_img)
    h, w = source_img.shape
    hh = len(canvas)
    ww = len(canvas)

    yoff = round((hh-h)/2)
    xoff = round((ww-w)/2)

    canvas[yoff:yoff+h, xoff:xoff+w] = source_img;

    return imutils.rotate(canvas, angle)

def get_h_masks(source_roimask):
    height = len(source_roimask)
    width = len(source_roimask[1])
    if width % 2 == 0:
        return (source_roimask[0:height, 0: (int(width / 2) - 1)], cv2.flip(source_roimask[0:height, (int(width / 2) + 0): (width - 1)], 1) ) 
    else:
        return (source_roimask[0:height, 0: (int(width / 2) - 1)], cv2.flip(source_roimask[0:height, (int(width / 2) + 1): (width - 1)], 1) ) # STUB  returns tuple


def compare_h_masks(h_mask_a, h_mask_b):
    pixel_count = 0
    pixel_delta = 0
    for i in range(len(h_mask_a)):
        for j in range(len(h_mask_a[i])):
            if h_mask_a[i][j] == 255 or h_mask_b[i][j] == 255:
                pixel_count += 1
            if h_mask_a[i][j] != h_mask_b[i][j]:
                pixel_delta += 1
    return (pixel_count, pixel_delta)

def calculate_theta(count, delta):
    return (count/100)*delta # NOTE deprecat

def calculate_theta2(count, delta):
    return (delta*100)/count

def calculate_roi_symmetry(source_img):
    # for no angle
    zero_h_mask_a, zero_h_mask_b = get_h_masks(source_img)
    zero_count, zero_delta = compare_h_masks(zero_h_mask_a, zero_h_mask_b)
    zero_theta = calculate_theta2(zero_count, zero_delta);

    best_theta = zero_theta
    best_theta_angle = 0
    
    for i in range(round(180/sym_precision)): 
        rotated_roi = get_rotated_roi(source_img, i)
        h_mask_a, h_mask_b = get_h_masks(rotated_roi)
        count, delta = compare_h_masks(h_mask_a, h_mask_b)
        theta = calculate_theta2(count,delta)
        if theta < best_theta:
            best_theta = theta
            best_theta_angle = i
        
    

    return best_theta, best_theta_angle

# t_theta, t_theta_angle = calculate_roi_symmetry(ROIGS[0])
# print(t_theta, t_theta_angle)
# cv2.imshow('rot2', get_rotated_roi(ROIGS[0], t_theta_angle))

ROI_SYM_THETA = []

for k in range(len(ROI)):
    theta, theta_angle = calculate_roi_symmetry(ROIMASK[k])
    print(theta, theta_angle)
    ROI_SYM_THETA.append(theta)
    rot =  get_rotated_roi(ROIGS[k], theta_angle)

task4 = np.zeros((dimensiuni[0], dimensiuni[1], 3), np.uint8)
task4[:] = bgbgr

for i in range(len(ROI)):
    x,y,w,h = cv2.boundingRect(contours[i])
    if(ROI_SYM_THETA[i] < sym_error_margin):
        pass
    else:
        task4 = cvzone.overlayPNG(task4, ROI[i],[x,y])

task5 = imagine.copy()

ROIAREA = []

for k in range(len(ROIMASK)):
    pixel_count = 0
    for i in range(len(ROIMASK[k])):
        for j in range(len(ROIMASK[k][i])):
            if ROIMASK[k][i][j] > 0:
                pixel_count+=1

    ROIAREA.append([k, pixel_count])

SORTED_ROIAREA = sorted(ROIAREA, key=lambda x: x[1], reverse=True)

print(SORTED_ROIAREA)

ROIARII=[]
for k in range(len(SORTED_ROIAREA)):
    for i in range(len(SORTED_ROIAREA)):
        index = SORTED_ROIAREA[k][0]
        x,y,w,h = cv2.boundingRect(contours[index])
        cv2.putText(task5, str(k+1), (x, y + round( h / 2 )), cv2.FONT_HERSHEY_SIMPLEX, 1, (122, 255, 0), 2, cv2.LINE_AA)

cv2.imshow("Task1", imagine)
cv2.imshow("Task2", task2)
cv2.imshow("Task3", task3)
cv2.imshow("Task4", task4)
cv2.imshow("Task5", task5)

cv2.waitKey(0)
cv2.destroyAllWindows()