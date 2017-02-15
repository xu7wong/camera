import numpy as np
import cv2
import random
X=400
Y=400
D=22#very important!
number=70 #very important!
img1= 255*np.ones([X,Y,3],dtype=np.uint8)
img2= 255*np.ones([X,Y,3],dtype=np.uint8)
#img2 = cv2.rectangle(img2,(2,2),(X+5,Y+5),(255,255,255),-1)
#img2 = cv2.circle(img2, (int((X)/2),int((Y)/2)), 200, (255, 255, 255), -1)
img2 = cv2.circle(img2, (int((X) / 2), int((Y) / 2)), 90, (0, 0, 0), -1)
gray1 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
mask1=gray1<255

img2= 255*np.ones([X,Y,3],dtype=np.uint8)
img2 = cv2.circle(img2, (int((X)/2),int((Y)/2)), int(X/2), (0, 0, 0), -1)
gray1 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
mask2=gray1>0
#k=np.where(gray1 >0)


for x in range(0, number-1):
	x_ran=int(random.uniform(0, X-1))
	y_ran = int(random.uniform(0, Y - 1))

	img1 = cv2.circle(img1,(x_ran,y_ran),D, (0,0,0),-1)
gray2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

while(1):

	gray1=gray2
	gray1[mask2] = 255
	gray1[mask1] = 255
	#img2 = cv2.circle(img2, (int((X + 8) / 2), int((Y + 8) / 2)), 200, (0, 0, 255), 1)
	#img2 = cv2.circle(img2, (int((X + 8) / 2), int((Y + 8) / 2)), 100-D, (255, 0, 255), 1)
	#img2[k[0],k[1]]=[255,255,255]
	cv2.imshow("PI", gray1)
	if cv2.waitKey(1) & 0xFF == ord('q'):

		cv2.imwrite( "g.jpg", gray2)
		break
cv2.destroyAllWindows()

