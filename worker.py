import numpy as np
import face_recognition
import json

from PIL import Image, ImageDraw
from math import hypot, sin, cos

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

def rotate_origin_only(xy, angle):
    if angle > 180:
        angle -= 360
        
    radians = angle / 180
    
    x, y = xy
    xx = x * cos(radians) + y * sin(radians)
    yy = -x * sin(radians) + y * cos(radians)

    return int(xx), int(yy)

def add_glasses(image, glasses_img, debug=False):
    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(np.array(image))
    
    if debug:
        print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

    glasses = Image.open(glasses_img)
    glasses_json = json.load(open(glasses_img + ".json"))

    # Cycle through each face
    for face_landmarks in face_landmarks_list:
        left_eye = face_landmarks["left_eye"]
        right_eye = face_landmarks["right_eye"]
        
        print(face_landmarks)
        
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
        
        if debug:
            d = ImageDraw.Draw(image)
            d.line([avg_left, avg_right], (255, 255, 255, 0), width=1)
            d.point(eyes_avg, (255 , 255, 0, 200))
        
    return image

def add_moustache(image, moustache_img, debug=False):
    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(np.array(image))
    
    if debug:
        print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

    moustache = Image.open(moustache_img)
    moustache_json = json.load(open(moustache_img + ".json"))

    # Cycle through each face
    for face_landmarks in face_landmarks_list:
        top_lip = face_landmarks["top_lip"]
        
        # Get lip angle - if face is rotated
        lip_angle = get_angle(top_lip[0], top_lip[6])
        
        # Get size of lip
        lip_sizex = dist_ab(top_lip[0], top_lip[6])
        
        # Rotate the moustache
        moustache = moustache.rotate(-lip_angle, expand=True)
        
        # Calculate the scaling
        x_scale_ratio = moustache.size[0] / lip_sizex / moustache_json["scale"]
        
        # Resize the glasses
        moustache = moustache.resize((int(moustache.size[0] / x_scale_ratio), 
                                      int(moustache.size[1] / x_scale_ratio)))
        
        avg_pos = get_average_pos(face_landmarks["nose_tip"] + face_landmarks["top_lip"])
        
        offset = rotate_origin_only((moustache_json["offset_x"], moustache_json["offset_y"]), lip_angle)

        moustache_paste = (avg_pos[0] - moustache.size[0] // 2 + offset[0], 
                           avg_pos[1] - moustache.size[1] // 2 + offset[1])
        
        image.paste(moustache, moustache_paste, moustache)
        
        if debug:
            for l in face_landmarks.keys():
                ImageDraw.Draw(image).polygon(face_landmarks[l])
        
    return image

def add_hat(image, hat_img, debug=False):
    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(np.array(image))
    
    if debug:
        print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

    hat = Image.open(hat_img)
    hat_json = json.load(open(hat_img + ".json"))

    # Cycle through each face
    for face_landmarks in face_landmarks_list:
        chin = face_landmarks["chin"]
        
        # Get angle - if face is rotated
        chin_angle = get_angle(chin[0], chin[-1])
        
        # Get size
        chin_sizex = dist_ab(chin[0], chin[-1])
        
        # Rotate the moustache
        hat = hat.rotate(-chin_angle, expand=True)
        
        # Calculate the scaling
        x_scale_ratio = hat.size[0] / chin_sizex / hat_json["scale"]
        
        # Resize the glasses
        hat = hat.resize((int(hat.size[0] / x_scale_ratio), 
                          int(hat.size[1] / x_scale_ratio)))
        
        avg_pos = get_average_pos((chin[0], chin[-1]))
        print(-chin_angle)
        
        offset = rotate_origin_only((hat_json["offset_x"], hat_json["offset_y"]), -chin_angle)

        moustache_paste = (avg_pos[0] - hat.size[0] // 2 + offset[0], 
                           avg_pos[1] - hat.size[1] // 2 + offset[1])
        
        image.paste(hat, moustache_paste, hat)
        
        if debug:
            ImageDraw.Draw(image).polygon(face_landmarks["chin"])
        
    return image

def get_polygon(src, polygon):
    # convert to numpy (for convenience)
    
    # create mask
   # mask = Image.new('L', (image.size[0], image.size[1]))
    #ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
     
    # assemble new image (uint8: 0-255)
    #newImArray = np.empty(imArray.shape,dtype='uint8')
#     
#     # colors (three first columns, RGB)
#     newImArray[:,:,:3] = imArray[:,:,:3]
#     
#     # transparency (4th column)
#     newImArray[:,:,3] = mask*255

    #image.paste(mask, (0, 0), mask)
    
    return src

def big_eyes(image):
    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(np.array(image))

    # Cycle through each face
    for face_landmarks in face_landmarks_list:
        #ImageDraw.Draw(image).polygon(face_landmarks["left_eye"])
        
        
        return get_polygon(image, face_landmarks["left_eye"])
        
        
        