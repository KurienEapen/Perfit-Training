import argparse
import logging
import time
import socket
import cv2
import numpy as np
import msvcrt
import math
import matplotlib.pyplot as plt 
import sqlite3

# Just disables the tensorflow warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

#variable 
ax = ay = bx = by = 0
ax2 = ay2 = bx2 = by2 = 0

def sqlite_setup(path=None):
	conn = sqlite3.connect(path)
	print(sqlite3.version)
	
	conn.execute('PRAGMA journal_mode=PERSIST')
	conn.execute('CREATE TABLE IF NOT exists  angle (points text)')
	conn.execute('insert into angle values (1)')
	conn.commit()
	return conn

def bearing(a1,  a2,  b1,  b2) :
	TWOPI = 6.2831853071795865
	RAD2DEG = 57.2957795130823209
	#if (a1 = b1 and a2 = b2) throw an error 
	theta = math.atan2(b1 - a1, a2 - b2)
	if (theta < 0.0):
		theta += TWOPI
	value = RAD2DEG * theta
	return "{:.2f}".format(value)
	#print (value)



if __name__ == '__main__':
	print("Starting")
	a1 = a2 = b1 = b2 = 0
	parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
	parser.add_argument('--camera', type=int, default=0)
	parser.add_argument('--zoom', type=float, default=1.0)
	parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
	parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
	parser.add_argument('--show-process', type=bool, default=False,
						help='for debug purpose, if enabled, speed for inference is dropped.')
	args = parser.parse_args()
	
	skeleton_db_conn = sqlite_setup("F://Unity 2017/Login/Assets/angle.s3db")
	skeleton_db_cur = skeleton_db_conn.cursor()
	sql = ''' UPDATE angle SET points = ?'''
	#total_time=0
	w, h = model_wh(args.resolution)
	e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
	cam = cv2.VideoCapture(args.camera)
	#cv2.namedWindow("test")
	total_time =0
	
	while (True):
		print("Running")
		flag = 0
		if msvcrt.kbhit():
			if ord(msvcrt.getch()) == 27:
				break
		skel_points = "-|"
		ret_val, image = cam.read()
		humans = e.inference(image)
		implot = plt.imshow(image)
		if len(humans) > 0:
			human = humans[0]
			for body_part in range(0, 17):
				if body_part in human.body_parts:
					#print("Bodypart " + str(body_part) + ": " + str(human.body_parts[body_part].x) + " - " + str(human.body_parts[body_part].y))
					skel_points += str(round(human.body_parts[body_part].x, 2)) + "-" + \
								str(round(human.body_parts[body_part].y, 2)) + "|"

					if body_part == 2:
						ax = human.body_parts[body_part].x
						ay = human.body_parts[body_part].y

					elif body_part == 4:
						bx = human.body_parts[body_part].x
						by = human.body_parts[body_part].y

					if body_part == 5:
						ax2 = human.body_parts[body_part].x
						ay2= human.body_parts[body_part].y

					elif body_part == 7:
						bx2 = human.body_parts[body_part].x
						by2 = human.body_parts[body_part].y
					# plt.plot(human.body_parts[body_part].x*640,human.body_parts[body_part].y*480,'o')
				else:
					skel_points += "-|"
		#print(skel_points)
		if (ax!=0 and ay !=0 and bx!=0 and by != 0):
			angleR = bearing (ay,ax,by,bx)
			if (angleR >= 300.0 and angleR <= 320.0):
				print ("RT TOP")
			elif(angleR >= 0.0 and angleR <= 40.0):
				print("RT MIDDLE")
			elif(angleR >= 65.0 and angleR<= 75.0):
				print("RT BOTTOM")
			elif(angleR>= 80.0 and angleR <= 95.0):
				print("RT INACTIVE")
		else :
			flag =1

		if (ax2!=0 and ay2 !=0 and bx2!=0 and by2 != 0):
			angleL = bearing (ay2,ax2,by2,bx2)
			if (angleL >= 250.0 and angleL <= 260.0):
				print ("LT TOP")
			elif(angleL >= 130.0 and angleL <= 160.0):
				print("LT MIDDLE")
			elif(angleL >= 110.0 and angleL<= 120.0):
				print("LT BOTTOM")
			elif(angleL>= 80.0 and angleL <= 100.0):
				print("LT INACTIVE")
		else :
			flag=1
		if flag == 0 :
			angles = str(angleR) + "|" + str(angleL)
		else :
			angles = "-"

		

		# plt.show()
		# cv2.waitKey(2)
		# cv2.destroyAllWindows()
		# plt.close()
		#cv2.imshow("test",image)
		
		#print (angle)
		skeleton_db_cur.execute(sql, [angles])
		skeleton_db_conn.commit()
	skeleton_db_conn.close()

	cv2.destroyAllWindows()
