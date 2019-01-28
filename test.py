import cv2
import numpy as np
import importlib
from PIL import Image

module = None
video_capture = cv2.VideoCapture(0)

while True:
    try:
        if module:
            importlib.reload(module)
        else:
            module = importlib.import_module("worker")
        
        # Grab a single frame of video
        ret, image_cam = video_capture.read()
        
        res = "res/hat/2.png"
        
        #image = module.add_hat(Image.fromarray(image_cam), res)
        image = module.big_eyes(Image.fromarray(image_cam).convert('RGB') )
    except:
        import traceback
        traceback.print_exc()
    
    cv2.imshow('Video', cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
    
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
