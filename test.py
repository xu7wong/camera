import cv2
import numpy as np
cap = cv2.VideoCapture(1)
cap.set(3,1280) # Width (in pixels)
cap.set(4,720) # Height (in pixels)
sift = cv2.xfeatures2d.SIFT_create()

x0=344
x1=214
x2=1041
x3=928

y0=39
y1=706
y2=706
y3=39



X=720
Y=720
Yr=417*3+10	#unit:mm
Xr=592*2	#unit:mm
scale=1.2	#img pixels vs. real unit_mm

pts1 = np.float32([[x0,y0],[x1,y1],[x2,y2],[x3,y3]])
pts2 = np.float32([[0,0],[0,Y],[X,Y],[Y,0]])
M_p = cv2.getPerspectiveTransform(pts1,pts2)

while(1):
	ret, frame = cap.read()
	gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	dst_temp = cv2.warpPerspective(gray2,M_p,(X,Y))
	#dst_temp=frame
	img2 = cv2.resize(dst_temp,(int(Xr/scale),int(Yr/scale)))
		
	kp2, des2 = sift.detectAndCompute(img2,None)
	
	print(len(img2))
	
	cv2.imshow("PI", img2)
	#print "image read OK!"	
	if cv2.waitKey(1) & 0xFF == ord('q'):
        	
		break
cap.release()
cv2.destroyAllWindows()

