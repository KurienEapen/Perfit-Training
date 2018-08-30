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
status = True
error1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0])
error2 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0])




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
	step1 = [0,0,0,1,0,0,0,0,0,0,0,0,1,1]
	step2 = [0,1,0,1,0,0,0,0,0,0,0,0,1,1]
	global error1,error2
	variance1 = np.array([abs(x) for x in (np.subtract(step1,ref))])
	variance2 = np.array([abs(x) for x in (np.subtract(step2,ref))])
	print("Variance1 :",variance1)
	print("Variance2 :",variance2)
	error1 += variance1
	error2 += variance2
	print("ref:",ref)
	print("error1 :",error1)
	print("error2 :",error2)
	return()



	

def check_loop(cam):

	global position_flag
	global position_reference
	global first_run_flag
	global previous_pos
	global status 
	while (True):
		print("Running")
		if msvcrt.kbhit():
			if ord(msvcrt.getch()) == 27:
				break
		skel_points = ""
		ret_val, image = cam.read()
		flag=0
		bodyparts_x=[]
		bodyparts_y=[]
		humans = e.inference(image)
		if len(humans) > 0:
			human = humans[0]
			for body_part in range(0, 18):
				if body_part in human.body_parts:
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
			print ("not detected")
			continue

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
		ref.append(int(Hip_R_Height > Knee_R))
		ref.append(int(Hand_L_Height > Spine_Mid))
		ref.append(int(Hand_L_Height > Head_Height))
		ref.append(int(Hand_R_Height > Head_Height))
		ref.append(int(Hand_L_Height > Hand_R_Height) if hand_difference >=0.03 else 0)
		ref.append(int(Hand_R_Height > Spine_Mid))
		ref.append(int(Hand_L_Height > Elbow_L_Height))
		ref.append(int(Hand_R_Height > Elbow_R_Height))
		ref.append(int(Knee_R > Knee_L) if knee_difference>= 0.03 else 0) 
		ref.append(int(Hand_L_Height > Knee_R))
		ref.append(int(Hand_R_Height > Knee_L))
		
		weight_generation(ref)

		
def error_cal(error):
	print("Error",error)
	total_weight_sum=np.sum(error)
	weight=np.array(np.subtract(total_weight_sum,error))
	print("first pass:",weight)
	weight=np.subtract(weight,min(weight))
	print("Inverted:",weight)
	total_weight_sum=np.sum(weight)
	weight=weight/total_weight_sum
	# weight=error/total_weight_sum
	# weight = np.array([abs(x) for x in (np.subtract(1,weight))])
	# total_weight_sum = np.sum(weight)
	# weight=weight/total_weight_sum
	print("Error weight :",weight)
	return weight


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

	weight1 = error_cal(error1)
	weight2 = error_cal(error2)

	cv2.destroyAllWindows()
#Continue loop till all useful points obtained




