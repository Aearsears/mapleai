import numpy as np
import time
import keyboard
import math
import threading

def attack_mob(boxes,classes):
    """
    recevies in the player box and the mob box and then will move the player towards the mob and then attack it
    """
    #midpoints X1 and X2
    player, closestmob = calculate_distance(boxes,classes)

    #vertical movement y axis
    if player[0]<closestmob[0]:
        keyboard.teledown()
    else:
        keyboard.teleup()

    # horizontal movement, i messed up the coordinates while creating the tuple index 1 is x, index 0 is y
    if player[1]<closestmob[1]:
        #moveleft and attack
        print("player coord:"+str(player[0])+" "+str(player[1]))
        print("\n mob coord:"+str(closestmob[0])+" "+str(closestmob[1]))
        keyboard.moveRight()
        keyboard.moveRight()
        # keyboard.moveRight()
        keyboard.attackFiveTimes()
        keyboard.loot()
    else:
        # mob is to the right and attack
        print("player coord:"+str(player[0])+" "+str(player[1]))
        print("\n mob coord:"+str(closestmob[0])+" "+str(closestmob[1]))
        keyboard.moveLeft()
        keyboard.moveLeft()
        # keyboard.moveLeft()
        keyboard.attackFiveTimes()
        keyboard.loot()


def filter(detections):
    """
   takes first five detections returns boxes,scores and classes as numpy arrays
    """
    #get first five predictions
    boxes = detections['detection_boxes'][0].numpy()[:5]
    scores = detections['detection_scores'][0].numpy()[:5]
    classes = (detections['detection_classes'][0].numpy() + 1).astype(int)[:5]

    isTherePlayer = False

    if 2 in classes[:]:
        isTherePlayer = True

    return boxes, scores, classes, isTherePlayer

def calculate_distance(boxes,classes):
    """
    calculates the distance between the player and the three mobs, and returns the mob with the shortest distance
    """
    #get the index of the player, returns a numpy array containing the index
    itemindex = np.where(classes==2)
    #get the midpoints, list of tuples
    midpoints =[]
    for i in range(np.shape(boxes)[0]):
        midpoints.append(getBoxesMidpoint(boxes[i]))
    
    #calculate the distance between the player and the mobs
    distance=np.zeros(5,dtype=np.float32)
    for i in range(np.shape(boxes)[0]):
        if i == itemindex[0][0]:
            distance[i]= 99999.0
        else:
            distance[i]=distance_2points(midpoints[i],midpoints[itemindex[0][0]])
    
    #get the min index, and return the player coord and mob coord.
    minindex = np.argmin(distance)
    return midpoints[itemindex[0][0]],midpoints[minindex]


def getBoxesMidpoint(box):
    """
    takes in normalized coordinates of the 800x600 screen. coordinates are xmin,ymin,xmax,ymax
    returns a tuple of the midpoint
    """
    #denormalize them
    normalized_coord = np.array([box[0]*806,box[1]*629,box[2]*806,box[3]*629],dtype=np.float32)
    #offset from the origin
    return (((normalized_coord[2]-normalized_coord[0])/2)+normalized_coord[0],((((normalized_coord[3]-normalized_coord[1])/2))+normalized_coord[1]))

def distance_2points(pt1,pt2):
    """
    returns distance between two points pt1(x1,y1),pt2(x2,y2). points as tuples. 
    """
    return math.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1])

def autobuff(stop_event):
    starttime = time.time()
    while not stop_event.wait(1):
        print("Buffing!")
        keyboard.buff()
        keyboard.buff()
        keyboard.buff()
        time.sleep(65.0 - ((time.time() - starttime) % 65.0))

def autocc(stop_event):
    starttime = time.time()
    while not stop_event.wait(1):
        print("CC'ing!")
        keyboard.cc()
        time.sleep(90.0 - ((time.time() - starttime) % 90.0))


if __name__ == "__main__":
    pass
    
    
