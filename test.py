from PIL import Image, ImageDraw
import numpy as np
import face_recognition
import json
import cv2
from math import hypot

def get_angle(p1, p2):
    dY = p1[1] - p2[1]
    dX = p1[0] - p2[0]
    return np.degrees(np.arctan2(dY, dX)) - 180

def dist_ab(p1,p2): 
    return hypot(p1[0] - p2[0], p1[1] - p2[1])

def avg_pos_rel_p1(p1, p2):
    return (
        p1[0] + (p2[0] - p1[0]) // 2, 
        p1[1] + (p2[1] - p1[1]) // 2)
    
def get_average_pos(array):
    avg = np.average(array, axis=0)
    return (int(avg[0]), int(avg[1]))

video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, image_cam = video_capture.read()
    
    image = Image.fromarray(image_cam)
    #image = image_cam
    
    glasses_img = "res/glasses.png"
    glasses_img_json = "res/glasses.png.json"
    
    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(np.array(image))
    
    print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))
    
    for face_landmarks in face_landmarks_list:
        # Print the location of each facial feature in this image
        facial_features = [
            'chin',
            'left_eyebrow',
            'right_eyebrow',
            'nose_bridge',
            'nose_tip',
            'left_eye',
            'right_eye',
            'top_lip',
            'bottom_lip'
        ]
        
        glasses = Image.open(glasses_img)
        glasses_json = json.load(open(glasses_img_json))
    
        left_eye = face_landmarks["left_eye"]
        right_eye = face_landmarks["right_eye"]
        
        # Get average positions for the eyes
        avg_left = get_average_pos(left_eye)
        avg_right = get_average_pos(right_eye)
    
        # Get eyes angle - if face is rotated
        eyes_angle = get_angle(avg_left, avg_right)
        
        # Get distance between eyes
        eyes_dist = dist_ab(avg_left, avg_right)
        
        # What's the center point between the eyes
        eyes_avg = avg_pos_rel_p1(avg_left, avg_right)
        
        # Rotate the glasses to match the angle of the eyes
        glasses = glasses.rotate(-eyes_angle, expand=True)
        
        # Calculate the scaling depending on the size of the eyes relative to the glasses
        x_scale_ratio = glasses.size[0] / eyes_dist / glasses_json["scale"]

        # Resize the glasses
        glasses = glasses.resize((int(glasses.size[0] / x_scale_ratio), int(glasses.size[1] / x_scale_ratio)))
        
        # Calculate where to paste the glasses
        glasses_paste = (eyes_avg[0] - glasses.size[0] // 2, eyes_avg[1] - glasses.size[1] // 2)

        image.paste(glasses, glasses_paste, glasses)
        
        # Debug
        #d = ImageDraw.Draw(image)
        #d.line([avg_left, avg_right], (255, 255, 255, 0), width=1)
        #glasses_coord = glasses.getbbox()
        #glasses_coord = (glasses_coord[0] + glasses_paste[0], glasses_coord[1] + glasses_paste[1], glasses_coord[2] + glasses_paste[0], glasses_coord[3] + glasses_paste[1])
        #d.point(eyes_avg, (255 , 255, 0, 200))
    
    cv2.imshow('Video', np.array(image))
    cv2.waitKey(1)
    
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
