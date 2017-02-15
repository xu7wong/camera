
#import curses
#stdscr = curses.initscr()
#stdscr.nodelay(1)
#curses.noecho()

import math
import cv2
import numpy as np
from matplotlib import pyplot as plt

import paho.mqtt.client as mqtt

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe("#")
def on_message(client, userdata, msg):
	pass
	print(msg.topic+" "+str(msg.payload))
#client = mqtt.Client('abdf123', True, None)
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
#client.connect("mqtt.3dprintingsystems.com", 1883, 60)
client.connect("139.59.233.107", 1886, 60)
client.loop_start()


cap = cv2.VideoCapture(1)
cap.set(3,1280) # Width (in pixels)
cap.set(4,720) # Height (in pixels)
ret, frame = cap.read()

"""
y0=226
y1=438
y2=y1
y3=y0
x0=259
x1=214
x2=505
x3=446
"""
x0=344
x1=223
x2=1041
x3=928

y0=51
y1=706
y2=706
y3=51



X=720
Y=720
Yr=417*3+10	#unit:mm
Xr=592*2	#unit:mm
scale=1.2	#img pixels vs. real unit_mm
flag=0
list_img_lib=['img/g1.jpg', 'img/g2.jpg', 'img/g3.jpg', 'img/g4.jpg','img/g5.jpg','img/g6.jpg']
list_robot = ['5C:CF:7F:24:89:37', '60:01:94:06:88:F1']
id_camera= 'camera/24:0A:64:63:9A:71'
list_img1 = [list_img_lib[4], list_img_lib[5]]


position_buffer=np.zeros([len(list_img1),3],dtype=np.uint64)

print(position_buffer)
X_camera=Xr/2	#unit:mm
Y_camera=Yr+50	#unit:mm
H_camera=1400	#unit:mm
H_robot=70	#unit:mm
areaA=int(Xr*Yr/scale/scale)

#frame= 255*np.ones([X,Y,3],dtype=np.uint8)

frame=cv2.line(frame,(x0,y0),(x1,y1),(255,0,0),4)
frame=cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),4)
frame=cv2.line(frame,(x2,y2),(x3,y3),(255,0,0),4)
frame=cv2.line(frame,(x3,y3),(x0,y0),(255,0,0),4)
frame = cv2.circle(frame,(100,100),15, (10,0,0),-1)
rows,cols,ch = frame.shape

pts1 = np.float32([[x0,y0],[x1,y1],[x2,y2],[x3,y3]])
pts2 = np.float32([[0,0],[0,Y],[X,Y],[Y,0]])

M_p = cv2.getPerspectiveTransform(pts1,pts2)

dst_temp = cv2.warpPerspective(frame,M_p,(X,Y))

#img2 = cv2.resize(dst_temp,None,fx=1.0, fy=Xr/Yr, interpolation = cv2.INTER_CUBIC)
img2 = cv2.resize(dst_temp,(int(Xr/scale),int(Yr/scale)))
#print len(img2)
#print len(img2[0])
plt.subplot(121),plt.imshow(frame),plt.title('Input')
plt.subplot(122),plt.imshow(img2),plt.title('Output')
plt.show()

MIN_MATCH_COUNT = 5
sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
list_kp=[]
list_des=[]

#print len(list_img1)
for x in range(0, len(list_img1)):
	print(list_img1[x])
	img1 = cv2.imread(list_img1[x])
	img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	kp1, des1 = sift.detectAndCompute(img1,None)
	list_kp.append(kp1)
	list_des.append(des1)

pts1 = np.float32([[x0,y0],[x1,y1],[x2,y2],[x3,y3]])
pts2 = np.float32([[0,0],[0,Y],[X,Y],[Y,0]])
M_p = cv2.getPerspectiveTransform(pts1,pts2)
#frame= 255*np.ones([720,1280,3],dtype=np.uint8)
ret, raw = cap.read()
frame=raw
while(ret):
	#"""
	#c = stdscr.getch()
	#if c == 113:
			
		#curses.nocbreak()
		#curses.echo()
		#curses.endwin()
		#break
	#"""
	
	gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
	try:
		
		dst_temp = cv2.warpPerspective(gray2,M_p,(X,Y))
		img2 = cv2.resize(dst_temp,(int(Xr/scale),int(Yr/scale)))

		kp2, des2 = sift.detectAndCompute(img2,None)
		for x in range(0, len(list_img1)):

			matches = flann.knnMatch(list_des[x],des2,k=2)
			good = []
			for m,n in matches:
				if m.distance < 0.6*n.distance:
					good.append(m)

			#print("key points: "+str(len(good)))

			if len(good)>MIN_MATCH_COUNT:
				#print "Start *******"
				print("Flag="+str(x)+" key points: "+str(len(good)))
				src_pts = np.float32([ list_kp[x][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
				dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

				M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
				matchesMask = mask.ravel().tolist()

				h,w = img1.shape
				pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)


				dst = cv2.perspectiveTransform(pts,M)
				img2 = cv2.polylines(img2,[np.int32(dst)],True, (10,0,0),1, cv2.LINE_AA)
				a=[np.int32(dst)]



				l1=math.sqrt(math.pow(a[0][0][0][0]-a[0][1][0][0],2)+math.pow(a[0][0][0][1]-a[0][1][0][1],2))
				l2=math.sqrt(math.pow(a[0][1][0][0]-a[0][2][0][0],2)+math.pow(a[0][1][0][1]-a[0][2][0][1],2))
				l3=math.sqrt(math.pow(a[0][2][0][0]-a[0][3][0][0],2)+math.pow(a[0][2][0][1]-a[0][3][0][1],2))
				l4=math.sqrt(math.pow(a[0][3][0][0]-a[0][0][0][0],2)+math.pow(a[0][3][0][1]-a[0][0][0][1],2))
				l0=math.sqrt(math.pow(a[0][0][0][0]-a[0][2][0][0],2)+math.pow(a[0][0][0][1]-a[0][2][0][1],2))
				p1=(l1+l2+l0)/2
				p2=(l3+l4+l0)/2
				#area2=[(a+b+c)(a+b-c)(a+c-b)(b+c-a)]^(1/2)
				area2=math.sqrt(p1*(p1-l0)*(p1-l2)*(p1-l1))+math.sqrt(p2*(p2-l0)*(p2-l3)*(p2-l4))

				#img2=cv2.drawKeypoints(img2,kp1,img2)

				#print area2
				#print areaA

				if area2<areaA*1.1 and area2>area2*0.9:



					X2= (a[0][0][0][0]+a[0][1][0][0]+a[0][2][0][0]+a[0][3][0][0])/4*scale
					Y2= (a[0][0][0][1]+a[0][1][0][1]+a[0][2][0][1]+a[0][3][0][1])/4*scale
					angle = 180-math.atan2((a[0][0][0][0]+a[0][3][0][0])/2  - (a[0][1][0][0]+a[0][2][0][0])/2,
							(a[0][0][0][1]+a[0][3][0][1])/2  - (a[0][1][0][1]+a[0][2][0][1])/2) / math.pi * 180
					L_prj=math.sqrt((X_camera-X2)*(X_camera-X2)+(Y_camera-Y2)*(Y_camera-Y2))*H_robot/H_camera

					Y_fix=L_prj*(Y_camera-Y2)/math.sqrt((X_camera-X2)*(X_camera-X2)+(Y_camera-Y2)*(Y_camera-Y2))
					X_fix=L_prj*(X_camera-X2)/math.sqrt((X_camera-X2)*(X_camera-X2)+(Y_camera-Y2)*(Y_camera-Y2))

					X_fixed=X2+X_fix-Xr/2
					Y_fixed=Y2+Y_fix-Yr/2
					img2 = cv2.putText(img2, str(x)+"a:"+str(int(angle)), (a[0][0][0][0], a[0][0][0][1]), 0, 0.7, (0, 0, 255), 2,
									   cv2.LINE_AA)
					#if int(abs(X_fixed-position_buffer[x,0]))<9 and int(abs(Y_fixed - position_buffer[x, 1]))<9 and int(abs(angle - position_buffer[x, 2])):
					client.publish(str(id_camera), "{\"robot_id_cam\":\""+str(list_robot[x])+"\",\"X\":" + str(int(X_fixed)) + ",\"Y\":"+ str(int(Y_fixed)) + ",\"A\":" + str(int(angle)) + "}")
					#else:
						#print("robot "+ str(x)+ " is not moving")


					#if abs(X_fixed-position_buffer[x,0])<3 and abs(Y_fixed-position_buffer[x,1])<3 and abs(angle-position_buffer[x,3])<3:
						#client.publish(str(list_robot[x]), "{\"X\":" + str(int(X_fixed)) + ",\"Y\":"
								#+ str(int(Y_fixed)) + ",\"A\":" + str(int(angle)) + "}")
					#client.publish("camera/60:01:94:06:88:F1", "{\"X\":" + str(int(X_fixed)) + ",\"Y\":"
								#+ str(int(Y_fixed)) + ",\"A\":" + str(int(angle)) + "}")
					#if flag==2:
						#client.publish("camera/60:01:94:06:88:F1", "{\"X\":" + str(int(X2)) + ",\"Y\":"
								#+ str(int(Y2)) + ",\"A\":" + str(int(angle)) + "}")
					position_buffer[x] = [int(X_fixed), int(Y_fixed), int(angle)]
					#print(position_buffer)
				else:
					print("Shape Error!")

			else:
				print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
				matchesMask = None

	except Exception:
		print("Program Error!")
		pass


	imgD = cv2.resize(img2,(int(Yr/scale/2),int(Xr/scale/2)))
	#frame= 255*np.ones([720,1280,3],dtype=np.uint8)

	cv2.imshow("PI", imgD)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	ret, raw = cap.read()
	frame = raw
cap.release()
cv2.destroyAllWindows()
