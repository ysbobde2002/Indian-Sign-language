import cv2
import numpy as np
import os

# Train or test 
mode = 'train'
directory = 'data1/'+mode+'/'

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Getting count of existing images
    count = {'zero': len(os.listdir(directory+"/0")),
             'one': len(os.listdir(directory+"/1")),
             'two': len(os.listdir(directory+"/2")),
             'three': len(os.listdir(directory+"/3")),
             'four': len(os.listdir(directory+"/4")),
             'five': len(os.listdir(directory+"/5")),
             'six': len(os.listdir(directory+"/6")),
             'seven': len(os.listdir(directory+"/7")),
             'eight': len(os.listdir(directory+"/8")),
             'nine': len(os.listdir(directory+"/9")),
             'a':len(os.listdir(directory+"/a")),
             'b':len(os.listdir(directory+"/b")),
             'c':len(os.listdir(directory+"/c")),
             'd':len(os.listdir(directory+"/d")),
             'e':len(os.listdir(directory+"/e")),
             'f':len(os.listdir(directory+"/f")),
             'g':len(os.listdir(directory+"/g")),
             'h':len(os.listdir(directory+"/h")),
             'i':len(os.listdir(directory+"/i"))}
            #  'j':len(os.listdir(directory+"/j"))}
    
    
    # Printing to the screen
    cv2.putText(frame, "WELCOME TO DE-SIGN-PHER ", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "MODE : "+mode, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "IMAGE COUNT", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ZERO : "+str(count['zero']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ONE : "+str(count['one']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "TWO : "+str(count['two']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "THREE : "+str(count['three']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "FOUR : "+str(count['four']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "FIVE : "+str(count['five']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "SIX : "+str(count['six']), (10, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "SEVEN : "+str(count['seven']), (10, 260), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "EIGHT: "+str(count['eight']), (10, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "NINE : "+str(count['nine']), (10, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "A : "+str(count['a']), (130, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "B : "+str(count['b']), (130, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "C : "+str(count['c']), (130, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "D : "+str(count['d']), (130, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "E : "+str(count['e']), (130, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "F : "+str(count['f']), (130, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "G : "+str(count['g']), (130, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "H : "+str(count['h']), (130, 260), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "I : "+str(count['i']), (130, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "J : "+str(count['j']), (130, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    
    # Region of Hand Image:
    # Coordinates for the frame
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Displaying the Hand Image window
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (64, 64)) 
 
    cv2.imshow("Frame", frame)
    
   # Converting Image to gray scale 
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("ROI", roi)
    #Getting the input till esc key is pressed and capturing the images accordingly.
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
    if interrupt & 0xFF == ord('0'):
        cv2.imwrite(directory+'0/'+str(count['zero'])+'.jpg', roi)
    if interrupt & 0xFF == ord('1'):
        cv2.imwrite(directory+'1/'+str(count['one'])+'.jpg', roi)
    if interrupt & 0xFF == ord('2'):
        cv2.imwrite(directory+'2/'+str(count['two'])+'.jpg', roi)
    if interrupt & 0xFF == ord('3'):
        cv2.imwrite(directory+'3/'+str(count['three'])+'.jpg', roi)
    if interrupt & 0xFF == ord('4'):
        cv2.imwrite(directory+'4/'+str(count['four'])+'.jpg', roi)
    if interrupt & 0xFF == ord('5'):
        cv2.imwrite(directory+'5/'+str(count['five'])+'.jpg', roi)
    if interrupt & 0xFF == ord('6'):
        cv2.imwrite(directory+'6/'+str(count['six'])+'.jpg', roi)
    if interrupt & 0xFF == ord('7'):
        cv2.imwrite(directory+'7/'+str(count['seven'])+'.jpg', roi)
    if interrupt & 0xFF == ord('8'):
        cv2.imwrite(directory+'8/'+str(count['eight'])+'.jpg', roi)
    if interrupt & 0xFF == ord('9'):
        cv2.imwrite(directory+'9/'+str(count['nine'])+'.jpg', roi)            
    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(directory+'a/'+str(count['a'])+'.jpg', roi)
    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(directory+'b/'+str(count['b'])+'.jpg', roi)
    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(directory+'c/'+str(count['c'])+'.jpg', roi)
    if interrupt & 0xFF == ord('d'):
        cv2.imwrite(directory+'d/'+str(count['d'])+'.jpg', roi)
    if interrupt & 0xFF == ord('e'):
        cv2.imwrite(directory+'e/'+str(count['e'])+'.jpg', roi)
    if interrupt & 0xFF == ord('f'):
        cv2.imwrite(directory+'f/'+str(count['f'])+'.jpg', roi)
    if interrupt & 0xFF == ord('g'):
        cv2.imwrite(directory+'g/'+str(count['g'])+'.jpg', roi)
    if interrupt & 0xFF == ord('h'):
        cv2.imwrite(directory+'h/'+str(count['h'])+'.jpg', roi)
    if interrupt & 0xFF == ord('i'):
        cv2.imwrite(directory+'i/'+str(count['i'])+'.jpg', roi)
    # if interrupt & 0xFF == ord('j'):
    #     cv2.imwrite(directory+'j/'+str(count['j'])+'.jpg', roi)
    
cap.release()
cv2.destroyAllWindows()

