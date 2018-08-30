import argparse
import logging
import time
import socket
import cv2
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import math
# Just disables the tensorflow warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

#Assuming subject is grounded

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

def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)

def jog_check(jog):
	bitmap_jogging = [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1],[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1]]
	acquired_bits= bin(int(''.join(map(str, jog)), 2) << 1)
	print(acquired_bits)
	reference_bits=bin(int(''.join(map(str, bitmap_jogging[0])), 2) << 1)
	print(reference_bits)
	y = ~(int(acquired_bits, 2)^int(reference_bits,2))
	print (bin(y)[2:].zfill(len(acquired_bits)))
	result=[int(d) for d in str(bin(y))[2:]]
	weights = [[0],[0],[.33],[0],[0],[0],[0],[0],[0],[.33],[.33],[0],[0],[0]]
	per_match = np.dot(result,weights)
	print(per_match)
	exit()


d=-1	


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

	
	#skeleton_db_conn = sqlite_setup("C:\\Perfit\\")
	#skeleton_db_cur = skeleton_db_conn.cursor()
	#sql = ''' UPDATE skeleton SET points = ?'''
	
	w, h = model_wh(args.resolution)
	e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
	cam = cv2.VideoCapture(args.camera)
	bodyparts_x=[]
	bodyparts_y=[]

    
	while True:
		print("Running")
		skel_points = ""
		d=d+1
		ret_val, image = cam.read()
		fn="images/pic%d.jpg"%d
		cv2.imwrite(fn,image)
		flag=0
		bodyparts_x=[]
		bodyparts_y=[]
		humans = e.inference(image)
		if len(humans) > 0:
			human = humans[0]
			for body_part in range(0, 18):
				if body_part in human.body_parts:
					print("Bodypart " + str(body_part) + ": " + str(human.body_parts[body_part].x) + " , " + str(human.body_parts[body_part].y))
					skel_points += str(body_part) + ": " + str(round(human.body_parts[body_part].x, 2)) + "," + \
								str(round(human.body_parts[body_part].y, 2)) + "; "
					bodyparts_x.append(round(human.body_parts[body_part].x,2))
					bodyparts_y.append(round(human.body_parts[body_part].y,2))				
				else:
					flag=1
					break
		if flag == 1:
			continue

		print(skel_points)
		print(bodyparts_x)
		print(bodyparts_y)

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

		ref.append(int(Hand_L_Height > Hip_L_Height))
		ref.append(int(Hand_R_Height > Hip_R_Height))
		ref.append(int((person_height/.7) * foot_difference > 0.1))
		# ref.append(int(Knee_R - Knee_L > 0.1))
		# ref.append(int(Knee_L - Knee_R > 0.1))
		ref.append(int(Hip_R_Height > Knee_R))
		ref.append(int(Hand_L_Height > Spine_Mid))
		ref.append(int(Hand_L_Height > Head_Height))
		ref.append(int(Hand_R_Height > Head_Height))
		ref.append(int(Hand_L_Height > Hand_R_Height))
		ref.append(int(Hand_R_Height > Spine_Mid))
		# ref.append(int(Elbow_R_Height > Shoulder_R_Distance + 0.2))
		# ref.append(int(Elbow_L_Height > Shoulder_L_Distance	- 0.2))
		ref.append(int(Hand_L_Height > Elbow_L_Height))
		ref.append(int(Hand_R_Height > Elbow_R_Height))
		# ref.append(int(Elbow_R_Height > Elbow_L_Height + 0.1))
		ref.append(int(Knee_R > Knee_L))
		#ref.append(int(Hand_R_Height - Knee_R > 0.4))
		ref.append(int(Hand_L_Height > Knee_R))
		ref.append(int(Hand_R_Height > Knee_L))


		print(ref)

		jog_check(ref)


		#skeleton_db_cur.execute(sql, [skel_points])
		#skeleton_db_conn.commit()
	#skeleton_db_conn.close()

	cv2.destroyAllWindows()

#Continue loop till all useful points obtained

