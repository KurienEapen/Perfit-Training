import argparse
import logging
import time
import socket
import cv2
import numpy as np

# Just disables the tensorflow warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

val=0
fps_time = 0
UDP_IP_ADDRESS = "127.0.0.1"
UDP_PORT_NO = 3010
UNITY_PORT_NO = 3012

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

	w, h = model_wh(args.resolution)
	e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
	cam = cv2.VideoCapture(args.camera)

	clientSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	clientSock.bind((UDP_IP_ADDRESS, UDP_PORT_NO))
	while True:
		clientSock.sendto(bytearray("Hello World", 'utf-8'), (UDP_IP_ADDRESS, 3011))
	# while True:
	# 	data, addr = clientSock.recvfrom(1024)
	# 	print("data / addr : " + str(data) + " / " + str(addr))
	# 	skel_points = ""
	# 	# user_input = int(user_input.strip())
	# 	# print("User input " + user_input)
	# 	# if data == "0":
	# 	# 	serverSock.close()
	# 	# 	print("Breaking")
	# 	# 	break
	# 	if "1" == "1":
	# 		print("Detecting skeletons")
	# 		ret_val, image = cam.read()
	# 		humans = e.inference(image)
	# 		if len(humans) > 0:
	# 			human = humans[0]
	# 			# for body_part in human.body_parts:
	# 			for body_part in range(0, 17):
	# 				if body_part in human.body_parts:
	# 					# print("Bodypart " + str(body_part) + ": " + str(human.body_parts[body_part].x) + " , " + str(human.body_parts[body_part].y))
	# 					skel_points += str(round(human.body_parts[body_part].x, 2)) + "," + \
	# 								str(round(human.body_parts[body_part].y, 2)) + ";"
	# 				else:
	# 					skel_points += ",;"
	# 			# print(human.body_parts[0].x + "," + human.body_parts[0].y)
	# 		clientSock.sendto(bytearray(skel_points, 'utf-8'), addr)
	cv2.destroyAllWindows()
