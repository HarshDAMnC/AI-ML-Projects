import google.generativeai as genai
import mediapipe as mp
import time
import cv2


hands=mp.solutions.hands #load the mediapipe hands module
hand=hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
#only 1 hand, no still but moving video

# Initialize the webcam, 0 is the default camera
cap = cv2.VideoCapture(0)

while True:
   
    success,img=cap.read() #read the image from webcam
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #convert the image to RGB bcz mediapipe works with RGB images
    result=hand.process(img)#process the image to detect hands(return true or false)
    fingercount=0

    if result.multi_hand_landmarks:#if hand is detected
        hand1=result.multi_hand_landmarks[0] #get the first hand detected
        landmark_list=[]
        h,w,_=img.shape #get the height and width of the image
        #hand1.landmark gives us the coordinates of the landmarks of the hand detected. hand1.landmark=[(x0,y0),....,(x20,y20)] where x and y are normalized coordinates between 0 and 1
        for idx in hand1.landmark:
            cx,cy=int(idx.x*w),int(idx.y*h)#convert coordinates from normalized to pixel values, for eg. if w=640, and coordinate of index tip is (0.5, 0.5), then cx=0.5*640=320, cy=0.5*480=240
            landmark_list.append((cx,cy)) #append the coordinates of the landmarks to the list
        for i in range(1,5):
            if landmark_list[(4*i)+4][1]<landmark_list[(4*i)+2][1]:
                fingercount+=1 #if tip is above joint, increament
        #checks if it is right or left hand
        if landmark_list[19][0]>landmark_list[11][0]:
            if landmark_list[4][0]<landmark_list[3][0]:
                fingercount+=1 #if thumb is not extended, then add 1 to the finger count
        else:
            if landmark_list[4][0]>landmark_list[3][0]:
                fingercount+=1
        time.sleep(1)
        print(f"Number of fingers detected: {fingercount}") #print the number of fingers detected
   
    """
                    (8)        (12)       (16)       (20)
                 |           |          |           |
                 |           |          |           |
        (5)-----(6)-----(10)-(11)--(14)-(15)--(18)--(19)
         |           |          |           |
        (4)         (9)        (13)        (17)
         |
        (3)
         |
        (2)
         |
        (1)
         |
        (0) ← wrist (base of palm)
         |
        Thumb
"""

cap.release()
cv2.destroyAllWindows()











