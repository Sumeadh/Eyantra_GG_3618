'''
*****************************************************************************************
*
*        		===============================================
*           		Geo Guide (GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 6 of Geo Guide (GG) Theme (eYRC 2023-24).
*  
*****************************************************************************************
'''

'''
* Team ID           : GG_3618
* Author List       : Logithsurya M A, Satyak R, Sumeadh M S, Sherwin Kumar M 
* Filename          : task_6.py
* Theme             : Geoguide
* Functions         : send_data,dijkstra, route,send_path_events,detect_corners,detect_ArUco_details,read_csv,write_csv, tracker,dict_coords,geo_location,detect,classify_event,save_events,update,find_out,put_text,detect_events,stop_at_events
'''

####################### IMPORT MODULES #######################
import numpy as np
import cv2
from cv2 import aruco
import csv
import time
from scipy.interpolate import CloughTocher2DInterpolator
import bluetooth
from queue import PriorityQueue
import numpy as np
import os 
import torch
from torchvision import transforms
from PIL import Image
import cv2 
import numpy as np                
import time   
##############################################################

####################### GLOBAL VARIABLES #######################
csv_name = "lat_long.csv"
aruco_id = 100
dict_coordinate ={}
vert_list_1 = [42,41,40,39,35,38,37,36,34,32,33,30,31,25,26,27,28,29]   # Aruco IDs list  for geolocation using interpolation
vert_list_2 = [48,47,46,45,44,46]

event_order = []                                                                                               # To store event order
order = []                                                                                                          #To get the final order without none
current = 'x0'                                                                                                  #Target node updates after reaching each node
device_address = "B0:A7:32:2B:72:8E "
paths = []                                                                                                         # Collection of all paths from Djikstra function
all_dir = []                                                                                                        #To Get the direction string which to be sent to the bot
nodes = {'x0': (63, 420), 'x1':(63, 377), 'x2':(63, 296), 'x3':(63, 204), 'x4':(63, 134), 'x5':(234, 134),'x6':(234, 204), 'x7':(234, 296), 'x8':(234, 377), 'x9':(410, 296), 'x10':(410, 204), 'x11': (410, 134), 'A':(122, 377), 'B':(331, 296), 'C':(330, 204), 'D':(118, 204), 'E': (100, 63)} #Nodes Location
maze={'x0':{'x1':44},
      'x1':{'x0':44,'x2':79,'A':53},
      'x2':{'x1':79,'x7':170,'x3':97.8},
      'x3':{'x4':62,'D':57,'x2':97.8},
      'x4':{'E':120,'x3':62,'x5':172},
      'x5':{'x4': 172,'x6':66,'x11':177},
      'x6':{'x5':66,'D':117,'C':97,'x7':81},
      'x7':{'x8':90,'B':98,'x2':170,'x6':81},
      'x8':{'A':109,'x9':246.248,'x7':90},
      'x9':{'B':80,'x10':86,'x8':246},
      'x10':{'C':76,'x11':66,'x9':86},
      'x11':{'x5':177,'E':330,'x10':66},
      'A':{'x1':53,'x8':109},
      'B':{'x7':98,'x9':80},
      'C':{'x10':76,'x6':97},
      'D':{'x3':57,'x6':117},
      'E':{'x4':120,'x11':330}}                                                                          #Distance between adjacent nodes
prev = None                                                                                                    #Previous Node in Djkstra algo
    
event_index = 0                                                                                              #Visited event number 
event_dict = {'A':[[110,230],[557,610]],'B':[[455,563],[414,460]],'C':[[440,575],[265,315]], 'D':[[70,200],[260,304]],'E':[[32,235],[21,113]],'END':[[27,83],[565,653]]}
#bounding_box for stop at events

combat = "combat"
rehab = "humanitarianaid"
military_vehicles = "militaryvehicles"
fire = "fire"
destroyed_building = "destroyedbuilding"
out_features = 5
input_path = "/home/logith/Downloads/EYRC_GG_3618/Task_6/"
input_path_2 = "/home/logith/Downloads/EYRC_GG_3618/Task_6/"
epochs = 10
trained_model = torch.load('model3.pth')
trained_model.eval()
event_list = []
event_list_1 = []
alpha = 1.15 # Contrast control
beta = 1.75
warp = np.zeros([700,700,3],dtype=np.uint8)
count = 0

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
data_transform = {
    'testing': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize]
    )
}

identified_labels = {"A":"None", "B":"None", "C": "None", "D": "None", "E" : "None"} 
r11 = (130, 80)
r12 = (205, 150)            
r21 = (110, 320)
r22 = (185, 395)            
r31 = (120, 615)
r32 = (200, 690)            
r41 = (465, 395)
r42 = (545, 325)            
r51 = (460, 470)
r52 = (530, 540)

###############################################################

###################### FUNCTION DEFINITION #######################

def send_data(device_address, data):           # To send the directions to the bot via BT
    sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    sock.connect((device_address, 1))
    sock.send(data)
    sock.close()

def dijkstra(maze,start,end,): #To find shortest path
    global prev
    end_corr=nodes[end]
    open=PriorityQueue()# To Find the shortest node
    came_from={}#To trace out the shortest path
    h_score = {i: float("inf") for i in maze}# Initially h_score of all nodes are infinite
    #print('h_score initially', h_score)
    path=[]
    open.put((0, start))
    open_hash=set()
    open_hash.add(start)
    current=None
    h_score[start]=0
    while current!=end:
        current=open.get()[1]
        neighbour_dict = maze[current]
        for neighbours in neighbour_dict:
            if neighbours!=prev:
                h_score_temp=h_score[current]+ neighbour_dict[neighbours]
                if  h_score_temp< h_score[neighbours]:
                    came_from[neighbours]=current #ex: C:B,B:A
                    h_score[neighbours]=h_score_temp
                if neighbours not in open_hash:
                    open.put((h_score_temp,neighbours))
                    open_hash.add(neighbours)
    while current in came_from:
        path.append(current)
        current=came_from[current]
    path = [start]+path[::-1]
    return path

def route(path): # To find the directions the bot must travel for the required node
    dir=[]
    dir.append('F')        
           
# Analyze the path and determine directions
    for i in range(len(path) - 2):  # Skip the last node as there is no next node for it
        current_node = path[i]
        next_node = path[i + 1]
        following_node = path[i + 2]

        # Get the coordinates of the nodes
        current_coordinates = nodes[current_node]
        next_coordinates = nodes[next_node]
        following_coordinates = nodes[following_node]

        # Calculate vectors representing the directions of the road segments
        vector1 = (next_coordinates[0] - current_coordinates[0], next_coordinates[1] - current_coordinates[1])
        vector2 = (following_coordinates[0] - next_coordinates[0], following_coordinates[1] - next_coordinates[1])

        # Calculate the cross product to determine the relative direction
        cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]

        if cross_product > 0:
            direction='R'
        elif cross_product ==0:
            direction='F' 
        elif cross_product<0:
            direction='L'

        dir.append(direction)
    return dir

def send_path_events(): #To send the path to reach between each events
    global current
    global order
    bt_path = []
    for items in event_order:
        if items != 'none':
            if items == "END":
                order.append('x0')
            else:
                order.append(items)
    for i in range(len(order)):
       path=dijkstra(maze,current,order[i])
       prev=path[-2]
       #Code to remove event node as intermediate nodes in path planning
       for j in range(1,len(path)-2):
        temp=[]
        if path[j] in order:
            temp.append(j)
        for index in temp:
            path.pop(index)
       paths.append(path)# appending path into total paths
       dir=route(path)
       all_dir.append(dir)
       current=order[i]
    
    for i in range(len(order)):  
        for j in range(len(all_dir[i])):
            bt_path.append(all_dir[i][j])
        bt_path.append('$')  
    bt_path.append('\n')
    for i in range(len(bt_path)):
        data=bt_path[i]
        send_data(device_address,data)
        time.sleep(0.1)

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
            (c11) = int(markerCorners[i][0][2][0])
            (c12) = int(markerCorners[i][0][2][1])
        if markerIds[i] == 4:
            c21 = int(markerCorners[i][0][3][0])
            c22 = int(markerCorners[i][0][3][1])
        if markerIds[i] == 6:
            c31 = int(markerCorners[i][0][0][0])
            c32 = int(markerCorners[i][0][0][1])
        if markerIds[i] == 7:
            c41 = int(markerCorners[i][0][1][0])
            c42 = int(markerCorners[i][0][1][1])
    if type(c11) == int and type(c21) == int and type(c31) == int and type(c41) == int:
        dst = np.array([[0, 0], [699, 0], [699, 699], [0, 699]], dtype = "float32")
        rect = np.array([[c11, c12], [c21, c22], [c31, c32], [c41, c42]], dtype = "float32")
        if rect is None:
            return(detect_corners(frame))

        else:
            print("Detected Corners")
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

def read_csv(csv_name):
    #Function to read the lat_lon.csv file and store the data in a dictionary
    lat_lon = {}
    with open(csv_name,'r') as csvfile:
        
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for rows in csvreader:
            key = int(rows[0])
            lat = float(rows[1])
            lon = float(rows [2])
            lat_lon[key] = [lat,lon]  
    return lat_lon

def write_csv(loc, csv_name):
     #Function to write in a csv file
    with open(csv_name ,'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['lat','lon'])
        csvwriter.writerow(loc)

def tracker(coordinate):
    write_csv(coordinate,"live_data.csv")

def dict_coords(filename,aruco_centres):
    #To map the aruco id centres to its lat and lon values
    coord = {}
    lat_lon = read_csv(filename)
    for keys in lat_lon.keys():
        if keys in aruco_centres.keys():
            if  keys == aruco_id:
                continue
            else:
                list1 = aruco_centres[keys]
                x=list1[0]
                y=list1[1]
                tuple1 = (x,y)
                coord[tuple1] = lat_lon[keys]
    return coord

def geo_location(latlon_dict,arucocentre,previous_centre):
    #To geolocate the bot using interpolation method and update csv file
    points = np.array(list(latlon_dict.keys()))
    latitudes, longitudes = zip(*[map(float, coord) for coord in latlon_dict.values()])
    previous_latitudes = previous_centre[0]
    previous_longitude = previous_centre[1]
    interp_lat = CloughTocher2DInterpolator(points, latitudes, fill_value = previous_latitudes)
    interp_lon = CloughTocher2DInterpolator(points, longitudes,fill_value = previous_longitude)

    lat = interp_lat(arucocentre)
    lon = interp_lon(arucocentre)
    coordinate = [str(round(lat[0],7)),str(round(lon[0],7))]
    
    tracker(coordinate)

def detect(label):
    if label == 0:
        return combat
    if label == 1:
        return destroyed_building
    if label == 2:
        return fire
    if label == 3:
        return rehab
    if label == 4:
        return military_vehicles
    
def classify_event(image):
    #Using the ml model to classify the placed events
    with torch.inference_mode():
        image = Image.open(image) 
        image = data_transform["testing"](image)
        image = image.unsqueeze(dim = 0)
        logit = trained_model(image)
        prob = torch.nn.functional.softmax(logit, 1)
        _, preds = torch.max(logit, 1)
        if prob[0][preds] <= 0.45:
            event = 'None'
        else:
            event = detect(preds)
    return event

def save_events(warp):
#To save the detected events
    a = warp[625:685, 127:190]
    b = warp[475:540, 465:530]
    c = warp[325:390, 470:545]
    d = warp[325:390, 115:180]
    e = warp[85:150, 130:195]
    cv2.imwrite("a.jpeg", a)
    cv2.imwrite("b.jpeg", b)
    cv2.imwrite("c.jpeg", c)
    cv2.imwrite("d.jpeg", d)
    cv2.imwrite("e.jpeg", e)

def update(identified_labels, event, answer):
    #To update the dictionary identified_labels
    global count
    count = count + 1
    if event == 'a':
        identified_labels.update({"A":answer})
    elif event == 'b':
        identified_labels.update({"B":answer})
    elif event == 'c':
        identified_labels.update({"C":answer})
    elif event == 'd':
        identified_labels.update({"D":answer})
    elif event == 'e':
        identified_labels.update({"E":answer})
    if count==5:
        time.sleep(1)

def find_out(identified_labels):
#Preprocessing
    for images in os.listdir(input_path):     
     if (images.endswith(".jpeg")):
        event_list_1.append(images)
        img = input_path + images        
        event_list.append(img)
    
    for i in range(5):
        image = cv2.imread(event_list[i])
        image= cv2.resize(image, (224, 224))
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  
        image = cv2.filter2D(image, -1, kernel)
        image = cv2.GaussianBlur(image, (3, 3), 1.0)
        sharpened = float(1.0 + 1) * image - float(1.0) * image
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        cv2.imwrite(event_list_1[i], sharpened)

    for event in event_list:
        answer = classify_event(event)
        update(identified_labels, event_list_1[event_list.index(event)][0], answer)

def put_text(identified_labels, warp): 
#Putting text over the video frame
    for key, value in identified_labels.items():
        if key == "A":
            cv2.putText(warp, value, r31, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if key == "B":
            cv2.putText(warp, value, r51, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if key == "C":
            cv2.putText(warp, value, r41, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if key == "D":
            cv2.putText(warp, value, r21, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
        if key == "E":
            cv2.putText(warp, value, r11, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.rectangle(warp, r11, r12, (0, 255, 0), 3) 
    cv2.rectangle(warp, r21, r22, (0, 255, 0), 3) 
    cv2.rectangle(warp, r31, r32, (0, 255, 0), 3) 
    cv2.rectangle(warp, r41, r42, (0, 255, 0), 3) 
    cv2.rectangle(warp, r51,r52, (0, 255, 0), 3)

    return warp

def detect_events(warp_frame):
#Compilation of all the functions
    save_events(warp_frame)
    find_out(identified_labels)
    put_text(identified_labels, warp_frame)
    print(identified_labels)
    events = ['none', 'none','none', 'none','none']
    place = ['none', 'none', 'none', 'none', 'none']
    x = 0
    for key, value in identified_labels.items():                
        if value == 'fire':
            place[0] = key
        if value == 'destroyedbuilding':
            place[1] = key
        if value == 'humanitarianaid':
            place[2] = key
        if value == 'combat':
            place[3] = key
        if value == 'miliaryvehicles':
            place[4] = key
    for i in range(5):
        if place[i] != 'none':
            events[x] = place[i]
            x = x + 1
    event_order.append(events[0])
    event_order.append(events[1])
    event_order.append(events[2])
    event_order.append(events[3])
    event_order.append(events[4])
    event_order.append('END')           
       
def stop_at_events(centre):
   # To make the bot stop at the events
   global event_index
   global order
   
   if (event_index < len(order)):  
        current_event = order [event_index]
    
        event_constraint = event_dict[current_event]
        
        xe1 = event_constraint [0][0]
        xe2 = event_constraint [0][1]
        ye1 = event_constraint [1][0]
        ye2 = event_constraint [1][1] 

        bot_x = centre[0]
        bot_y = centre[1] 
        
        if (current_event == 'x0'):
            if (xe1<bot_x<xe2) and (ye1 < bot_y <ye2):
                print("Visited Event :", order[event_index])
                send_data(device_address,'Q')
                event_index = event_index +1
        else:  
            if (xe1<bot_x<xe2) and (ye1 < bot_y <ye2):
                print("Visited Event :", order[event_index])
                send_data(device_address,'S')
                event_index = event_index +1
        

################################################################
if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    lat_lon = read_csv('lat_long.csv')
    previous_centre = [0,0]
    dst = []
    rect = []

    while True:
        ret, frame = cap.read()  
        #cv2.imshow("Video",frame)
        rect,dst = detect_corners(frame)
        if rect is not None:
            break

    while True:
        ret, frame = cap.read()
        time.sleep(1)
        M = cv2.getPerspectiveTransform(rect, dst)
        warp_frame = cv2.warpPerspective(frame, M, (700, 700))
        ArUco_details_dict = detect_ArUco_details(warp_frame)
        if (len(ArUco_details_dict)==48):
            print("Detected All Arucos")
            for keys in ArUco_details_dict.keys():
                if keys in vert_list_1:
                    value = ArUco_details_dict[keys]
                    value_1 = [value[0],(value[1]-29)]
                    ArUco_details_dict[keys] = value_1
                elif keys in vert_list_2:
                    value = ArUco_details_dict[keys]
                    value_1 = [value[0],(value[1]-95)]
                    ArUco_details_dict[keys] = value_1
                else:
                    continue
            dict_coordinate = (dict_coords(csv_name,ArUco_details_dict))
            detect_events(warp_frame)
            send_path_events()
            break
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        clock = 0
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        M = cv2.getPerspectiveTransform(rect, dst)
        warp_frame = cv2.warpPerspective(frame, M, (700, 700))
        warp_frame = cv2.convertScaleAbs(warp_frame, alpha=alpha, beta=beta)
        

        centres = detect_ArUco_details(warp_frame)
    
        if aruco_id in centres.keys():
            centre = centres.get(aruco_id)
            cv2.circle(warp_frame,centre, 4, (0,0,255), -1)
            previous_centre = centre
        else:
            centre = previous_centre

        geo_location(dict_coordinate,centre,previous_centre)
        stop_at_events(centre)

        warp_frame = put_text(identified_labels, warp_frame)
        
        cv2.imshow("Output",warp_frame)
        if cv2.waitKey(1) == ord('q'):
            break
        # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

