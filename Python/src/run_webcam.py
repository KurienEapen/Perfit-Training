import argparse
import logging
import time

import cv2
import numpy as np

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
val=0
fps_time = 0


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
	parser.add_argument('--camera', type=int, default=0)
	parser.add_argument('--zoom', type=float, default=1.0)
	parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
	parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
	parser.add_argument('--show-process', type=bool, default=False,
						help='for debug purpose, if enabled, speed for inference is dropped.')
	args = parser.parse_args()

	logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
	w, h = model_wh(args.resolution)
	e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
	logger.debug('cam read+')
	cam = cv2.VideoCapture(args.camera)
	ret_val, image = cam.read()
	logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

	print('waiting for user input')
	while True:
		user_input=input()
		user_input=int(user_input)
		print(user_input)
		if user_input == 0:
			print("Breaking")
			break
        # #sys.argv[-1]
		if user_input==1:
			print('Loop running')
			ret_val, image = cam.read()
			humans = e.inference(image)
			#cv2.waitKey(1000)
			val=0


			image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

			logger.debug('show+')
			cv2.putText(image,
			                "FPS: %f" % (1.0 / (time.time() - fps_time)),
			                (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
			                (0, 255, 0), 2)
			#cv2.imshow('tf-pose-estimation result', image)
			fps_time = time.time()
			#val=0
			if cv2.waitKey(1) == 27:
			    break
			logger.debug('finished+')
			print('val- ',val)
		
	cv2.destroyAllWindows()
