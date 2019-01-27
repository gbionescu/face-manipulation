from PIL import Image, ImageDraw
import numpy as np
import face_recognition
import json
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
    
def load_json(fname):
    json_parsed = {}
    json_data = None
    
    with open(fname) as f:
        json_data = json.load(f)
        
    json_this = {}
    for el, val in json_data["this"].items():
        split = val.replace("(", "").replace(")", "").split(",")
        json_this[el] = (int(split[0]), int(split[1]))
        
    json_target = {}
    for el, val in json_data["target"].items():
        json_target[el] = val
        
    json_parsed["this"] = json_this
    json_parsed["target"] = json_target

    return json_parsed

def get_average_pos(array):
    avg = np.average(array, axis=0)
    return (int(avg[0]), int(avg[1]))

# Load the jpg file into a numpy array
#image = Image.open("examples/biden.jpg")
image = Image.open("res/asd.jpg")
image = image.convert("RGB")

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
    glasses_json = load_json(glasses_img_json)
    print(glasses_json)

    left_eye = face_landmarks["left_eye"]
    right_eye = face_landmarks["right_eye"]
    
    avg_left = get_average_pos(left_eye)
    avg_right = get_average_pos(right_eye)

    # Get eyes angle, distance and center point
    eyes_angle = get_angle(avg_left, avg_right)
    eyes_dist = dist_ab(avg_left, avg_right)
    eyes_avg = avg_pos_rel_p1(avg_left, avg_right)
    
    glasses = glasses.rotate(-eyes_angle, expand=True)
    glsize = glasses.size
    
    x_scale_ratio = glsize[0] / eyes_dist / 1.9
    
    glasses = glasses.resize((int(glsize[0] / x_scale_ratio), int(glsize[1] / x_scale_ratio)))
    
    glsize = glasses.size
    glasses_paste = (eyes_avg[0] - glsize[0] // 2, eyes_avg[1] - glsize[1] // 2)
    image.paste(glasses, glasses_paste, glasses)
    
    d = ImageDraw.Draw(image)

    #d.line([avg_left, avg_right], (255, 255, 255, 0), width=1)

    glasses_coord = glasses.getbbox()
    glasses_coord = (glasses_coord[0] + glasses_paste[0], glasses_coord[1] + glasses_paste[1], glasses_coord[2] + glasses_paste[0], glasses_coord[3] + glasses_paste[1])
    
    d.point(eyes_avg, (255 , 255, 0, 200))

image.show()
