#!/usr/bin/env python
import argparse
import logging
import time
import socket
import cv2
import numpy as np
import sqlite3
#import pymysql
import time
import msvcrt

# Just disables the tensorflow warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh


def sqlite_setup(path=None):
	# db_path = os.path.join(path, "skeleton.db")
	# if os.path.exists(path):
	#  	os.remove(path)
	#  	os.remove(path+".meta")
	
	try:
		conn = sqlite3.connect(path)
		print(sqlite3.version)
	except Error as e:
		print(e)
	finally:	
		conn.execute('PRAGMA journal_mode=PERSIST')
		conn.execute('CREATE TABLE IF NOT exists  skeleton (points text)')
		conn.execute('insert into skeleton values (1)')
		conn.commit()
	return conn


if __name__ == '__main__':
	print("Starting")
	parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
	parser.add_argument('--camera', type=int, default=0)
	parser.add_argument('--zoom', type=float, default=1.0)
	parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
	parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
	parser.add_argument('--show-process', type=bool, default=False,
						help='for debug purpose, if enabled, speed for inference is dropped.')
	args = parser.parse_args()
	
	skeleton_db_conn = sqlite_setup("F://Unity 2017/Login/Assets/skeleton.s3db")
	skeleton_db_cur = skeleton_db_conn.cursor()
	sql = ''' UPDATE skeleton SET points = ?'''
	w, h = model_wh(args.resolution)
	e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
	cam = cv2.VideoCapture(args.camera)
	
	#total_time = 0;
	while(True):#while(True):
		print("Running")
		if msvcrt.kbhit():
			if ord(msvcrt.getch()) == 27:
				break
		bodypart_flag=0
		skel_points = "|"
		ret_val, image = cam.read()
		# start = time.time()
		humans = e.inference(image)
		# end = time.time()
		# total_time = total_time +(end - start) 
		#print("time: ",end - start)
		#print(timeit.timeit(humans))
		if len(humans) > 0:
			human = humans[0]
			for body_part in [2,4,5,7]:
				if body_part in human.body_parts:
					#print("Bodypart " + str(body_part) + ": " + str(human.body_parts[body_part].x) + " , " + str(human.body_parts[body_part].y))
					skel_points += str("{:.2f}".format(round(human.body_parts[body_part].x, 2))) + "-" + \
								str("{:.2f}".format(round(human.body_parts[body_part].y, 2))) + "|"
					bodypart_flag += 1
				
		print(skel_points)
		if bodypart_flag==4:
			skeleton_db_cur.execute(sql, [skel_points])
		
		skeleton_db_conn.commit()
	#avg_time = total_time/20
	#print ("average time : ",avg_time)
	#time.sleep(3)
	skeleton_db_cur.execute(''' SELECT points FROM skeleton''')
	data = skeleton_db_cur.fetchone()
	print(data)
	skeleton_db_conn.close()
	cv2.destroyAllWindows()
