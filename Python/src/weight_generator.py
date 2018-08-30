import argparse
import logging
import time
import socket
import cv2
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import sys
import msvcrt


# Just disables the tensorflow warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
from PIL import Image


#Assuming subject is grounded
position_reference=[]
position_flag = 0
first_run_flag = 0
previous_pos = []
variation = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0])
#d=-1



def sqlite_setup(path=None):
	db_path = os.path.join(path, "skeleton.db")
	if os.path.exists(db_path):
		os.remove(db_path)
	conn = sqlite3.connect(db_path)
	conn.execute('PRAGMA journal_mode=wal')
	conn.execute('CREATE TABLE skeleton (points text)')
	conn.execute('insert into skeleton values (1)')
	conn.commit()
	return conn


def weight_generation(ref):
	global previous_pos 
	global variation
	variance = np.array([abs(x) for x in (np.subtract(previous_pos,ref))])
	print(variance)
	variation += variance
	print(variation)
	previous_pos = ref
	return()



	

def check_loop(cam):

	global position_flag
	global position_reference
	global first_run_flag
	global previous_pos
	for i in range(20):
		print("Running")
		skel_points = ""
		#d=d+1
		ret_val, image = cam.read()
		#fn="images/pic%d.jpg"%d
		#cv2.imwrite(fn,image)
		flag=0
		bodyparts_x=[]
		bodyparts_y=[]
		humans = e.inference(image)
		if len(humans) > 0:
			human = humans[0]
			for body_part in range(0, 18):
				if body_part in human.body_parts:
					#print("Bodypart " + str(body_part) + ": " + str(human.body_parts[body_part].x) + " , " + str(human.body_parts[body_part].y))
					skel_points += str(body_part) + ": " + str(round(human.body_parts[body_part].x, 2)) + "," + \
								str(round(human.body_parts[body_part].y, 2)) + "; "
					bodyparts_x.append(round(human.body_parts[body_part].x,2))
					bodyparts_y.append(round(human.body_parts[body_part].y,2))				
				else:
					flag=1
					skel_points+="';"
					bodyparts_y.append("@")
					bodyparts_x.append("@")
		if flag == 1:
			print ("yesss")
			continue

		#print(skel_points)
		#print(bodyparts_x)
		#print(bodyparts_y)
		if not bodyparts_x or not bodyparts_y:
			continue
		ground = max(bodyparts_y[10],bodyparts_y[13])
		Head_Height = ground - bodyparts_y[0];
		SpineMid_Height = ground - bodyparts_y[1];
		Elbow_L_Height = ground - bodyparts_y[6];
		Elbow_R_Height = ground - bodyparts_y[3];
		Wrist_L_Height = ground - bodyparts_y[7];
		Wrist_R_Height = ground - bodyparts_y[4];
		Hip_L_Height = ground - bodyparts_y[11];
		Hip_R_Height = ground - bodyparts_y[8];
		Hand_L_Height = ground - bodyparts_y[7];
		Hand_R_Height = ground - bodyparts_y[4];
		Knee_R = ground - bodyparts_y[9];
		Knee_L = ground - bodyparts_y[12];

		Hand_R_Distance = ground - bodyparts_y[4];
		Hand_L_Distance = ground - bodyparts_y[7];
		Shoulder_L_Distance = bodyparts_x[5]-bodyparts_x[1];
		Shoulder_R_Distance = bodyparts_x[1]-bodyparts_x[2];

		Spine_Base = (bodyparts_y[8]+bodyparts_y[11])/2;
		Spine_Shoulder = ground - bodyparts_y[1];
		Spine_Mid = (Spine_Base + Spine_Shoulder)/2;


		ref=[]
		foot_difference=abs(bodyparts_y[10]-bodyparts_y[13])#between two feet
		person_height=abs(bodyparts_y[13]-bodyparts_y[14])#foot to head 
		hand_difference = abs (bodyparts_y[4]-bodyparts_y[7])#between two hands
		knee_difference = abs (bodyparts_y[9]-bodyparts_y[12])#between two knees

		
		ref.append(int(Hand_L_Height > Hip_L_Height))
		ref.append(int(Hand_R_Height > Hip_R_Height))
		ref.append(int((person_height/.7) * foot_difference > 0.1))
		# ref.append(int(Knee_R - Knee_L > 0.1))
		# ref.append(int(Knee_L - Knee_R > 0.1))
		ref.append(int(Hip_R_Height > Knee_R))
		ref.append(int(Hand_L_Height > Spine_Mid))
		ref.append(int(Hand_L_Height > Head_Height))
		ref.append(int(Hand_R_Height > Head_Height))
		ref.append(int(Hand_L_Height > Hand_R_Height) if hand_difference >=0.03 else 0)

		ref.append(int(Hand_R_Height > Spine_Mid))
		# ref.append(int(Elbow_R_Height > Shoulder_R_Distance + 0.2))
		# ref.append(int(Elbow_L_Height > Shoulder_L_Distance	- 0.2))
		ref.append(int(Hand_L_Height > Elbow_L_Height))
		ref.append(int(Hand_R_Height > Elbow_R_Height))
		# ref.append(int(Elbow_R_Height > Elbow_L_Height + 0.1))
		ref.append(int(Knee_R > Knee_L) if knee_difference>= 0.03 else 0) 
		#ref.append(int(Hand_R_Height - Knee_R > 0.4))
		ref.append(int(Hand_L_Height > Knee_R))
		ref.append(int(Hand_R_Height > Knee_L))

		if first_run_flag == 0 :
			first_run_flag = 1 
			previous_pos = ref
			continue

		elif first_run_flag == 1 :
			weight_generation(ref)

		if msvcrt.kbhit():
			if ord(msvcrt.getch()) == 27:
				break



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
	bodyparts_x=[]
	bodyparts_y=[]

    
	check_loop(cam)

	print(variation)
	total_weight_sum=np.sum(variation)
	print(total_weight_sum)
	weighted_result=variation/total_weight_sum
	print(weighted_result)

	cv2.destroyAllWindows()

#Continue loop till all useful points obtained

