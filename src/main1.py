
import cv2
import numpy as np
import math
import sys
import pyscreenshot as ImageGrab

ipab=0
def draw_previous():
	for (x,y,w,h) in old_boxes:
		cv2.circle(frame,(getCentroid(x,y,w,h)),3,(255,0,255),3)

def getCentroid(x,y,w,h):
	return x+int(w/2),y+int(h/2)

def getCentroids(boxes):
	centroid = []
	for (x,y,w,h) in boxes:centroid.append(getCentroid(x,y,w,h))
	return centroid

def distance(point_1,point_2):
	return (math.sqrt( (point_2[0] - point_1[0])**2 + (point_2[1] - point_1[1])**2 ))

def see_distance(old_centroids,new_centroids):
	i = 0
	distances = []
	while i < len(old_centroids) and i < len(new_centroids):
		distances.append(distance(old_centroids[i],new_centroids[i]))
		i+=1
	print(distances)

def get_closest(p1,new_centroids):
	minimum , index = sys.maxsize , 0
	for i in range(len(new_centroids)):
		d = distance(p1,new_centroids[i])
		if d <= minimum:
			minimum = d
			index = i
	if index == 0 and len(new_centroids) == 0:new_centroids.append(p1)
	return index , minimum , new_centroids[index]

def check(height,count_cars,count_bikes,old_centroids,new_centroids,contours,threshold,area):
	for p1 in old_centroids:
		if p1[1] >= height:
			index , distance , p2 = get_closest(p1,new_centroids)
			if distance > threshold:continue
			if p2[1] < height:
				if cv2.contourArea(contours[index]) > area:
					cam.read()

					count_cars += 1
				else:count_bikes += 1
				print("\r",end="")
				continue
		if p1[1] <= height:
			index , distance , p2 = get_closest(p1,new_centroids)
			if distance > threshold:continue
			if p2[1] > height:
				if cv2.contourArea(contours[index]) > area:
					count_cars += 1
				else:count_bikes += 1
				print("\r",end="")
	return count_cars , count_bikes


def drawBoundingBox(opening,frame,count_cars,count_bikes,min_length,width,height):
	global old_boxes , new_boxes
	_,contours,hierarchy = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	old_boxes , new_boxes , cnts = new_boxes[:] , [] , []
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		if w < min_length or h < min_length: continue
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
		cx , cy = getCentroid(x,y,w,h)
		cv2.circle(frame,(cx,cy),3,(255,0,0),3)
		new_boxes.append((x,y,w,h))
		cnts.append(cnt)
	cv2.line(frame,(350,height),(width-200,height),(0,255,0),3)
	cv2.line(frame,(350,height-30),(width-200,height-30),(0,255,0),3)
	cv2.line(frame,(350,height+30),(width-200,height+30),(0,255,0),3)

	old_centroids , new_centroids = getCentroids(old_boxes) , getCentroids(new_boxes)
	count_cars , count_bikes = check(height,count_cars,count_bikes,old_centroids,new_centroids,cnts,40,12500)
	return count_cars , count_bikes , frame


def getBackground(cam,count,step):
	ret ,frame = cam.read()
	avg = np.float32(frame)
	for i in range(1,count):
		for j in range(step):ret ,frame = cam.read()
		cv2.accumulateWeighted(frame,avg,0.01)
		res = cv2.convertScaleAbs(avg)
		print("\r",end="")
		print("Initialising",int(i/count*100),"%",end="")
	print("\rInitialising 100 %")
	background = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
	cv2.imshow("final_frame",frame)
	return background

# Adding the results to the output
def display_final(frame,size,mask,count,x1,y1,x2,y2):
	roi = frame[x1:size[0]+x1,y1:size[1]+y1]
	bg = cv2.bitwise_and(roi,roi,mask=mask)
	frame[x1:size[0]+x1,y1:size[1]+y1] = bg
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(frame,str(count),(x2,y2), font, 2,(0,0,0),3)
	return frame

source = '../videos/c2.avi'
cam = cv2.VideoCapture(source)
background = getBackground(cam,300,2)
cv2.imshow('background',background)

cam = cv2.VideoCapture(source)

kernel1 = np.ones((3,3),np.uint8)
kernel2 = np.ones((5,5),np.uint8)

width , height = background.shape[1] , background.shape[0]
new_boxes , old_boxes ,count_cars,count_bikes = [] , [] , 0 , 0


car = cv2.imread('car.png',0)
bike = cv2.imread('bike.png',0)
size_car = car.shape
bike = cv2.resize(bike,(int(bike.shape[0]/2),int(bike.shape[1]/2)))
size_bike = bike.shape
ret , mask_car = cv2.threshold(car,220,255,cv2.THRESH_BINARY)
ret , mask_bike = cv2.threshold(bike,220,255,cv2.THRESH_BINARY)

while(1):
	ret ,frame = cam.read()
	if not ret:break

	frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	frame_diff = cv2.absdiff(background,frame_gray)
	ret , thresh = cv2.threshold(frame_diff,35,255,cv2.THRESH_BINARY)
	erosion = cv2.erode(thresh, kernel1, iterations=1)
	dilation = cv2.dilate(erosion,kernel2 , iterations=2)
	median = cv2.medianBlur(dilation,5)

	count_cars, count_bikes , final = drawBoundingBox(median,frame,count_cars,count_bikes,30,width,int(height/2))
	cv2.imshow('framediff',frame_diff)
	#cv2.imshow('median',median)
	cv2.imshow('thresh',thresh)
	#cv2.imshow('erosion',erosion)
	cv2.imshow('dilate',dilation)
	final = display_final(frame,size_car,mask_car,count_cars,0,0,220,80)
	final = display_final(final,size_bike,mask_bike,count_bikes,0,350,500,80)
	cv2.namedWindow('final',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('final', 1920,1080)
	cv2.imshow('final',final)
	k = cv2.waitKey(30) & 0xFF
	if k == ord('q'):break

print()
cv2.destroyAllWindows()
cam.release()
