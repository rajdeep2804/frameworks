import pandas as pd
import json
import numpy as np
import cv2


data=pd.read_csv('card_data.csv')


def genrate_pts(poly_line):
    res = json.loads(poly_line)
    print(res)
    coordinates = []
    for j in range(len(res)):
        a = res[j]['x']
        b = res[j]['y']
        c = [int(a) , int(b)]
        coordinates.append(c)
    pts = np.array(coordinates)
    return pts

def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)

    return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]


def ployline_test(data):
    for i in range(len(data)):

        poly_line = data["polygon"][i]
        image_name = data["image_name"][i]   
        print(image_name)

        image_path = 'card_images/'+image_name
        image = cv2.imread(image_path)
        pts = genrate_pts(poly_line)
        pts = pts.reshape((-1, 1, 2))
        isClosed = True
        color = (255, 0, 0)
        thickness = 2
        # Using cv2.polylines() method
        # Draw a Blue polygon with 
        # thickness of 1 px
        image = cv2.polylines(image, [pts], isClosed, color, thickness)
        filename = 'output_polyline/output_'+image_name
        print("filename : ", filename)
        cv2.imwrite(filename, image)
    
def bbox_test(data):
    for i in range(len(data)):

        poly_line = data["polygon"][i]
        image_name = data["image_name"][i]   
        print(image_name)

        image_path = 'card_images/'+image_name
        image = cv2.imread(image_path)
        pts = genrate_pts(poly_line)
        topleft, bottomright = bounding_box(pts)
        cv2.rectangle(image, topleft, bottomright, (255, 0, 0), 4)
        filename = 'output_bbox/output_'+image_name
        print("filename : ", filename)
        cv2.imwrite(filename, image)    
bbox_test(data)