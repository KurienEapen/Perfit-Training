#!/usr/bin/env python
import argparse
import logging
import time
import socket
import cv2
import numpy as np
import sqlite3
import pymysql
import requests

# Just disables the tensorflow warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh



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
	
	# db = pymysql.connect("localhost","u934734608_user","password","u934734608_db" )
	# cursor = db.cursor()
	# sql = ''' UPDATE skeleton SET points = ?'''
	# cursor.execute('insert into skeleton values (1)')

	w, h = model_wh(args.resolution)
	e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
	cam = cv2.VideoCapture(args.camera)
	
	for bleh in range(10):
		print("Running")
		point_flag=0
		skel_points = "|"
		ret_val, image = cam.read()
		humans = e.inference(image)
		if len(humans) > 0:
			human = humans[0]
			for body_part in range(0, 18):
				if body_part in human.body_parts:
					print("Bodypart " + str(body_part) + ": " + str(human.body_parts[body_part].x) + " , " + str(human.body_parts[body_part].y))
					skel_points += str("{:.2f}".format(round(human.body_parts[body_part].x, 2))) + "-" + \
								str("{:.2f}".format(round(human.body_parts[body_part].y, 2))) + "|"
				else:
					skel_points += "-|"
					point_flag=1
		# cursor.execute(sql, [skel_points])
		
		# cursor.commit()
		post_data={"points": skel_points}
		if point_flag==0:
			resp=requests.post("https://radicals-ar-ed.000webhostapp.com/upload_points.php",params=post_data)
		print(skel_points)
		time.sleep(1)
	cv2.destroyAllWindows()
