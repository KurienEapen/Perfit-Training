from os import listdir
from os.path import isfile, join
import cv2
folder = "SlideShow"
images = [f for f in listdir(folder) if isfile(join(folder, f))]
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
for image in images:
    img = cv2.imread(folder+"/"+image)
    cv2.imshow('image',img)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        exit()
cv2.destroyAllWindows()