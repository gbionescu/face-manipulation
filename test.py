import cv2
import numpy as np
import importlib
import glob
from PIL import Image

module = None
video_capture = cv2.VideoCapture(0)


reslist = glob.glob("res/eyes/*.png")

idx = 0
while True:
    try:
        if module:
            importlib.reload(module)
        else:
            module = importlib.import_module("worker")

        # Grab a single frame of video
        ret, image_cam = video_capture.read()

        res = reslist[idx % len(reslist)]
        print(res)

        image = module.add_eyes(Image.fromarray(image_cam).convert('RGB'), res)
        idx += 1

    except:
        import traceback
        traceback.print_exc()

    cv2.imshow('Video', cv2.cvtColor(np.array(image), cv2.CV_8UC1))
    cv2.waitKey(1)

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
