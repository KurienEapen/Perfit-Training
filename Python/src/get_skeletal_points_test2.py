import argparse
import logging
import time
import socket
import cv2
import numpy as np
import sqlite3
import matplotlib.pyplot as plt

# Just disables the tensorflow warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
from PIL import Image

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

d=-1
picloc="images/jog4.jpg"
bitmap_jogging = [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1],[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1]]
print(bitmap_jogging)
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
	cam = cv2.imread(picloc)
	bpx=[]
	bpy=[]

    
	print("Running")
	skel_points = ""
	d=d+1
	
	bpx=[]
	bpy=[]
	#(ret_val, image) = cam.read()
	humans = e.inference(cam)
	if len(humans) > 0:
		human = humans[0]
		for body_part in range(0, 18):
			if body_part in human.body_parts:
				print("Bodypart " + str(body_part) + ": " + str(human.body_parts[body_part].x) + " , " + str(human.body_parts[body_part].y))
				skel_points += str(body_part) + ": " + str(round(human.body_parts[body_part].x, 2)) + "," + \
							str(round(human.body_parts[body_part].y, 2)) + "; "
				bpx.append(round(human.body_parts[body_part].x,2))
				bpy.append(round(human.body_parts[body_part].y,2))				
			else:

				skel_points += ",;"	
				bpx.append("@");
				bpy.append("@");
			
	print(skel_points)
	print(bpx)
	print(bpy)

		#skeleton_db_cur.execute(sql, [skel_points])
		#skeleton_db_conn.commit()
	#skeleton_db_conn.close()

	cv2.destroyAllWindows()

ground = max(bpy[10],bpy[13])
Head_Height = ground - bpy[0];
SpineMid_Height = ground - bpy[1];
Elbow_L_Height = ground - bpy[6];
Elbow_R_Height = ground - bpy[3];
Wrist_L_Height = ground - bpy[7];
Wrist_R_Height = ground - bpy[4];
Hip_L_Height = ground - bpy[11];
Hip_R_Height = ground - bpy[8];
Hand_L_Height = ground - bpy[7];
Hand_R_Height = ground - bpy[4];
Knee_R = ground - bpy[9];
Knee_L = ground - bpy[12];

Hand_R_Distance = ground - bpy[4];
Hand_L_Distance = ground - bpy[7];
Shoulder_L_Distance = bpx[5]-bpx[1];
Shoulder_R_Distance = bpx[1]-bpx[2];

Spine_Base = (bpy[8]+bpy[11])/2;
Spine_Shoulder = ground - bpy[1];
Spine_Mid = (Spine_Base + Spine_Shoulder)/2;

ref=[]
foot_difference=abs(bpy[10]-bpy[13])#between two feet
person_height=abs(bpy[13]-bpy[14])#foot to head 

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

imm=Image.open(picloc)
width, height = imm.size

im = plt.imread(picloc)
implot = plt.imshow(im)
plt.plot([h * width for h in bpx],[v * height for v in bpy],'o')
plt.show()