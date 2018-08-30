import argparse
import logging
import time
import ast

import common
import cv2
import numpy as np
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

from lifting.prob_model import Prob3dPose
from lifting.draw import plot_pose

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run') 						# Adding arguments to the programs
    parser.add_argument('--image', type=str, default='../images/p1.jpg')							# Adding images name else it will take the default image
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')	# Specify resolution 
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')			# Specify Model
    parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')	# Scales - Reason Unknown 
    args = parser.parse_args()												# Argument contain all the parse
    scales = ast.literal_eval(args.scales)

    w, h = model_wh(args.resolution) #Return width and height into w, h respectively after checking if its a multiple of 16
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))		# Model + width and height

    # estimate human poses from a single image !
    image = common.read_imgfile(args.image, None, None)
    # image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    t = time.time()
    humans = e.inference(image, scales=scales)
    elapsed = time.time() - t

    logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))
    print(humans)
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    # cv2.imshow('tf-pose-estimation result', image)
    # cv2.waitKey()

    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

   
