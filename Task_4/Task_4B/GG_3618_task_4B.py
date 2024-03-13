'''
*****************************************************************************************
*
*        		===============================================
*           		Geo Guide (GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 4B of Geo Guide (GG) Theme (eYRC 2023-24).
*  
*****************************************************************************************
'''

# Team ID:			GG_3618
# Author List:		Logithsurya M A, Satyak R, Sumeadh M S, Sherwin Kumar M 
# Filename:			task_4b.py
# Functions:		detect_corners, detect_ArUco_details, calculate_distance, distance_to_aruco, near_aruco_id, read_csv, write_csv,tracker(
		


####################### IMPORT MODULES #######################
import numpy as np
import cv2
from cv2 import aruco
import math
import csv
import time
##############################################################
csv_name = "lat_long.csv"


def detect_corners(frame):
    #Detecting the corners of the given map and applying a transform for the video frame
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)
    c11 = [0, 0]
    c21 = [0, 0]
    c31 = [0, 0]
    c41 = [0, 0]
    c12 = [0, 0]
    c22 = [0, 0]
    c32 = [0, 0]
    c42 = [0, 0]
    for i in range(len(markerIds)):
        if markerIds[i] == 5:
            # print("5")
            # print(markerCorners[i])
            (c11) = int(markerCorners[i][0][2][0])
            (c12) = int(markerCorners[i][0][2][1])
        if markerIds[i] == 4:
            # print("4")
            # print(markerCorners[i])
            c21 = int(markerCorners[i][0][3][0])
            c22 = int(markerCorners[i][0][3][1])
        if markerIds[i] == 6:
            # print("6")
            # print(markerCorners[i])
            c31 = int(markerCorners[i][0][0][0])
            c32 = int(markerCorners[i][0][0][1])
        if markerIds[i] == 7:
            # print("7")
            # print(markerCorners[i])
            c41 = int(markerCorners[i][0][1][0])
            c42 = int(markerCorners[i][0][1][1])
    if type(c11) == int and type(c21) == int and type(c31) == int and type(c41) == int:
        dst = np.array([[0, 0], [899, 0], [899, 899], [0, 899]], dtype = "float32")
        rect = np.array([[(c11), c12], [c21, c22], [c31, c32], [c41, c42]], dtype = "float32")
         
        return rect,dst
   
def detect_ArUco_details(image):
    #Detecting all the arucos present in the map
    arucoDict = cv2.aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    
    corners , ids, _ = aruco.detectMarkers(image,arucoDict)
    
    ArUco_centres = {}

    for i in range(len(ids)):
        #ArUco_corners[ids[i][0]] = corners[i][0]

        # Calculate center coordinates and orientation
        cx = int((corners[i][0][0][0] + corners[i][0][2][0]) / 2)
        cy = int((corners[i][0][0][1] + corners[i][0][2][1]) / 2)  
        centres=[cx,cy]
    
        ArUco_centres[int(ids[i])]= centres
   
    return ArUco_centres

def calculate_distance(point1, point2):
    # Function to calculate distace between two points
    x1, y1 = point1
    x2, y2 = point2
        # Rest of your code that uses x and y

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def distance_to_aruco(target_centre, Aruco_coord):
    # Function to calculate distance between the aruco id of the bot and all the id present on the map
    result_dist = {}
    for ids, centres in Aruco_coord.items():
        if ids == 1:
            continue
        else:
            distance = calculate_distance(target_centre, centres)
            result_dist[ids] = distance
    return result_dist

def near_aruco_id(dist_aruco):
    #Function to find the nearest aruco id to the bot
    if not dist_aruco:
        return None
    min_value = min(dist_aruco.values())
    for key, value in dist_aruco.items():
        if value == min_value:
            near_id=key
    return near_id

def read_csv(csv_name):
    #Function to read the lat_lon.csv file and store the data in a dictionary
    lat_lon = {}
    with open(csv_name,'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for rows in csvreader:
            key = rows[0]
            lat = rows[1]
            lon = rows [2]
            lat_lon[key] = [lat,lon]  
    return lat_lon

def write_csv(loc, csv_name):

    with open(csv_name ,'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['lat','lon'])
        csvwriter.writerow(loc)

def tracker(ar_id, lat_lon):

    coordinate = lat_lon.get(str(ar_id))
    write_csv(coordinate,"live_data.csv")

    return coordinate
	

if __name__ == "__main__":

    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    lat_lon = read_csv('lat_long.csv')
    previous_centre = [0,0]
    dst = []
    rect = []

    while True:
        ret, frame = cap.read()
        rect,dst = detect_corners(frame)
        if rect is not None:
            break

    while True:
        ret, frame = cap.read()
        M = cv2.getPerspectiveTransform(rect, dst)
        warp_frame = cv2.warpPerspective(frame, M, (900, 900))
        ArUco_details_dict = detect_ArUco_details(warp_frame)
        if (len(ArUco_details_dict)==48):
            print("Detected All Arucos")
            break
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warp_frame = cv2.warpPerspective(frame, M, (900, 900))
    
        centres = detect_ArUco_details(warp_frame)

        if 1 in centres.keys():
            centre = centres.get(1)
            cv2.circle(warp_frame,centre, 4, (0,0,255), -1)
            previous_centre = centre
        else:
            centre = previous_centre
        
        aruco_dist = distance_to_aruco(centre,ArUco_details_dict)

        near_id = near_aruco_id(aruco_dist)

        t_point = tracker(near_id,lat_lon)

        cv2.imshow("Output",warp_frame)

        if cv2.waitKey(1) == ord('q'):
            break
        # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

