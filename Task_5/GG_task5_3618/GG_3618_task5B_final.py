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
csv_name = "lat_long.csv"
aruco_id = 100
dict_coordinate ={}

vert_list_1 = [42,41,40,39,35,38,37,36,34,32,33,30,31,25,26,27,28,29]
vert_list_2 = [48,47,46,45,44,46]

device_address = "B0:A7:32:2B:72:8E"

checkpoint=['A','B','C','D','E','F','G','H','I','J']
directions=[]
paths=[]
global vazhi
vazhi=[]
count=0
nodes = {'x0': (63, 420), 'x1':(63, 377), 'x2':(63, 296), 'x3':(63, 204), 'x4':(63, 134), 'x5':(234, 134),'x6':(234, 204), 'x7':(234, 296), 'x8':(234, 377), 'x9':(410, 296), 'x10':(410, 204), 'x11': (410, 134), 'A':(122, 377), 'B':(331, 296), 'C':(310, 204), 'D':(118, 204), 'E': (100, 63)}
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
      'E':{'x4':120,'x11':330}}


event_order = []
event_index = 0
event_dict = {'A':[[110,230],[557,610]],'B':[[455,590],[414,460]],'C':[[415,575],[265,315]], 'D':[[70,200],[260,304]],'E':[[32,269],[21,113]],'END':[[27,67],[565,653]]}

combat = "combat"
rehab = "humanitarianaid"
military_vehicles = "militaryvehicles"
fire = "fire"
destroyed_building = "destroyedbuilding"
out_features = 5
input_path = "/home/logith/Downloads/EYRC_GG_3618/task 5a/"
input_path_2 = "/home/logith/Downloads/EYRC_GG_3618/task 5a/"
epochs = 10
trained_model = torch.load('model5.pth')
trained_model.eval()
event_list = []
event_list_1 = []
alpha = 1.15 # Contrast control
beta = 1.75
warp = np.zeros([700,700,3],dtype=np.uint8)

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
r41 = (470, 395)
r42 = (545, 325)            
r51 = (460, 470)
r52 = (530, 540)

def send_data(device_address, data):
    sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    sock.connect((device_address, 1))
    sock.send(data)
    sock.close()
    
def route(path):
  dir=[]
  if(count>=2):
    l=len(paths[count-2])
    if(paths[count-2][l-2]==path[1]):
        dir.append('U')
    else:
        dir.append('F')
  else:
     dir.append('F')        
           
# Analyze the path and determine directions
  for i in range(len(path) - 2):  # Skip the last node as there is no next node for it
    current_node = path[i]
    next_node = path[i + 1]
    following_node = path[i + 2]
    if(next_node=='A' or next_node=='B' or next_node=='C' or next_node=='D' or next_node=='E'):
        continue

    # Get the coordinates of the nodes
    current_coordinates = nodes[current_node]
    next_coordinates = nodes[next_node]
    following_coordinates = nodes[following_node]

    # Calculate vectors representing the directions of the road segments
    vector1 = (next_coordinates[0] - current_coordinates[0], next_coordinates[1] - current_coordinates[1])
    vector2 = (following_coordinates[0] - next_coordinates[0], following_coordinates[1] - next_coordinates[1])

    # Calculate the cross product to determine the relative direction
    cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]

    # Determine the direction (right or left)
    #direction = 'Right' if cross_product > 0 elif 'F' else 'Left'
    
    if (following_node=='E' or current_node=='E' ):
        direction='F'
    elif(current_node=='x7' and next_node=='x8' and following_node=='A') :
        direction='R'  
    elif(current_node=='x9' and next_node=='x8' and following_node=='A') :
        direction='F'        
    elif(current_node=='A' and next_node=='x8' and following_node=='x7') :
        direction='L'    
    elif(current_node=='B' and next_node=='x9' and following_node=='x8') :
        direction='R'  
    elif(current_node=='x10' and next_node=='x9' and following_node=='x8') :
        direction='F'            
    else:    
     if cross_product > 0:
        direction='R'
     elif cross_product ==0:
        direction='F' 
     elif cross_product<0:
        direction='L'

    dir.append(direction)
    #print(dir)
  directions.append(dir)
    
def dijkstra(maze,start,end,):
    global count
    count+=1
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
    paths.append(path)
    #print(paths)
    route(path)
    
def send_path_events():
  global vazhi
  """if(event_order[0]=='C' and event_order[1]=='D'):
        vazhi=['F','F','R','L','R','$','F','L','L','L','R','$','F','L','F','F','$','\n']
  else:
    dijkstra(maze,'x0',event1)
    lenn=len(paths[0])

    passed_node=paths[0][lenn-2]

    keylist=list(maze[event1].keys())

    keylist.remove(passed_node)

    dijkstra(maze,keylist[0],event2)

    middy=[event1,paths[1][0],paths[1][1]]
    route(middy)

    directions[1].insert(1,directions[2][1])
    directions.pop()

    passed_node2=paths[1][len(paths[1])-2]
    keylist2=list(maze[event2].keys())
    keylist2.remove(passed_node2)

    dijkstra(maze,keylist2[0],'x0')

    middy2=[event2,paths[2][0],paths[2][1]]
    route(middy2)

    directions[2].insert(1,directions[3][1])
    directions.pop()
    print(directions)
    for i in range(3):  
        for j in range(len(directions[i])):
            vazhi.append(directions[i][j])
        vazhi.append('$')  
    vazhi.append('\n')"""
  vazhi=['F', 'F', 'F', 'F', 'F', '$', 'F', 'F', 'F', 'R', '$', 'F', 'L', 'R', '$', 'F', 'R' ,'F', 'R', '$', 'F', 'F', '$', 'F', 'R', 'R', 'F', 'L','F', '$', '\n']
  print(vazhi)
  for i in range(len(vazhi)):
        data=vazhi[i]
        send_data(device_address,data)
        print(data)
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
        dst = np.array([[0, 0], [699, 0], [699, 699], [0, 699]], dtype = "float32")
        rect = np.array([[c11, c12], [c21, c22], [c31, c32], [c41, c42]], dtype = "float32")
        if rect is None:
            detect_corners(frame)
        else:
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

    with open(csv_name ,'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['lat','lon'])
        csvwriter.writerow(loc)

def tracker(coordinate):

    write_csv(coordinate,"live_data.csv")

def dict_coords(filename,aruco_centres):
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
    points = np.array(list(latlon_dict.keys()))
    latitudes, longitudes = zip(*[map(float, coord) for coord in latlon_dict.values()])
    previous_latitudes = previous_centre[0]
    previous_longitude = previous_centre[1]
    interp_lat = CloughTocher2DInterpolator(points, latitudes, fill_value = previous_latitudes)
    interp_lon = CloughTocher2DInterpolator(points, longitudes,fill_value = previous_longitude)

    lat = interp_lat(arucocentre)
    lon = interp_lon(arucocentre)
    coordinate = [lat[0],lon[0]]
    
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
  
    with torch.inference_mode():
        image = Image.open(image) 
        image = data_transform["testing"](image)
        image = image.unsqueeze(dim = 0)
        logit = trained_model(image)
        prob = torch.nn.functional.softmax(logit, 1)
        _, preds = torch.max(logit, 1)
        # print(prob)
        if prob[0][preds] <= 0.60:
            event = 'None'
        else:
            event = detect(preds)        
    return event

def save_events(warp):
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

def find_out(identified_labels):

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
    for key, value in identified_labels.items():
        if key == "A":
            cv2.putText(warp, value, r31, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if key == "B":
            cv2.putText(warp, value, r51, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if key == "C":
            R41 = (r41[0] + 80, r41[1])
            cv2.putText(warp, value, R41, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if key == "D":
            cv2.putText(warp, value, r21, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
        if key == "E":
            cv2.putText(warp, value, r11, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.rectangle(warp, r11, r12, (0, 255, 0), 3) 
    cv2.rectangle(warp, r21, r22, (0, 255, 0), 3) 
    cv2.rectangle(warp, r31, r32, (0, 255, 0), 3) 
    cv2.rectangle(warp, r41, r42, (0, 255, 0), 3) 
    cv2.rectangle(warp, r51,r52, (0, 255, 0), 3)
def stop_at_events(centre):
   global event_index
   if (event_index < len(event_order)):  
        current_event = event_order [event_index]
    
        event_constraint = event_dict[current_event]
    
        xe1 = event_constraint [0][0]
        xe2 = event_constraint [0][1]
        ye1 = event_constraint [1][0]
        ye2 = event_constraint [1][1]   
        
        bot_x = centre[0]
        bot_y = centre[1] 
        
        if (xe1<bot_x<xe2) and (ye1 < bot_y <ye2):
            print("Visited Event :", event_order[event_index])
            send_data(device_address,'S')
            event_index = event_index +1

def bottom_turn_left(centre):
    
        xe1 = 589
        xe2 = 616
        ye1 = 553
        ye2 = 604  
        bot_x = centre[0]
        bot_y = centre[1] 
        if (xe1<bot_x<xe2) and (ye1 < bot_y <ye2):
            # print("TURN LEFT")
            send_data(device_address,'S')
            what_turn='l'

def bottom_turn_right(centre):
    
        xe1 = 604
        xe2 = 656
        ye1 = 548
        ye2 = 570
        bot_x = centre[0]
        bot_y = centre[1] 
        
        if (xe1<bot_x<xe2) and (ye1 < bot_y <ye2):
            # print("TURN RIGHT")
            send_data(device_address,'S')
            what_turn='r'            

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
        #cv2.imshow("Video",frame)
        time.sleep(2)
        rect,dst = detect_corners(frame)
        if rect is not None:
            break

    while True:
        ret, frame = cap.read()
        time.sleep(1)
        M = cv2.getPerspectiveTransform(rect, dst)
        warp_frame = cv2.warpPerspective(frame, M, (700, 700))
        ArUco_details_dict = detect_ArUco_details(warp_frame)
        # print(len(ArUco_details_dict))
        if (len(ArUco_details_dict)==48):
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
            print("Detected All Arucos")
            save_events(warp_frame)
            #find_out(identified_labels)
            put_text(identified_labels, warp_frame)
            events = ['none', 'none', 'none', 'none', 'none']
            place = ['none', 'none', 'none', 'none', 'none']
            x = 0
            for key, value in identified_labels.items():                
                if value == 'fire':
                    place[0] = key
                if value == 'destroyedbuilding':
                    place[1] = key
                if value == 'humanitarianaid':
                    place[2] = key
                if value == 'militaryvehicles':
                    place[3] = key
                if value == 'combat':
                    place[4] = key
            for i in range(5):
                if place[i] != 'none':
                    event_order[x] = place[i]
                    x = x + 1
            
            event_order.append('END')

            """print(event_order[0], event_order[1])"""
            """cv2.imshow("output1", warp_frame)"""
            time.sleep(4)
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
        image = cv2.GaussianBlur(warp, (3, 3), 1.0)
        sharpened = float(1.0 + 1) * image - float(1.0) * image
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
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
        # bottom_turn_right(centre)
        # bottom_turn_left(centre)
        # cv2.rectangle(warp_frame,(604,548),(656,570),(0,0,255),2)
        # cv2.rectangle(warp_frame,(589,553),(616,604),(0,255,255),2)
        put_text(identified_labels, warp_frame)
        cv2.imshow("Output",warp_frame)
        if cv2.waitKey(1) == ord('q'):
            break
        # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

