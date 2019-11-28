from imutils import face_utils
import dlib
import cv2
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
import pickle

regressor_pkl_filename = 'decision_tree_10000.pkl'
regressor_pkl = open(regressor_pkl_filename, 'rb')
regressor = pickle.load(regressor_pkl)

# p = our pre-treined model directory, on my case, it's on the same script's diretory.
datPath = "codes/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(datPath)
class detectors:
    def getMinMaxScaling(self,filePath):
        df = pd.read_csv(filePath,header=None)
        del df[136]
        x = df.iloc[0,:].values.reshape(2,68)[0]
        y = df.iloc[0,:].values.reshape(2,68)[1]
        x = x.reshape(1,-1)
        y = y.reshape(1,-1)
        self.x_scaler = MinMaxScaler()
        self.x_scaler.fit(x)
        self.y_scaler = MinMaxScaler()
        self.y_scaler.fit(y)
        return

    def getKSSvalue(self,shape):
        print_shape = shape
        x = print_shape.reshape(2,68)[0].reshape(1,-1)
        y = print_shape.reshape(2,68)[1].reshape(1,-1)
        x = self.x_scaler.transform(x)
        y = self.y_scaler.transform(y)
        x = x.reshape(1,-1)
        y = y.reshape(1,-1)
        x_index = 0
        y_index = 0
        print_shape = print_shape.reshape(1,-1)
        for i in range(print_shape.shape[1]):
            if i%2 == 0:
                print_shape[0][i] =  x[0][x_index]
                x_index+=1
            else:
                print_shape[0][i] =  y[0][y_index]
                y_index+=1
            print_shape = print_shape.reshape(1,-1)
            print("Your drowsy Score is ",regressor.predict(print_shape)[0])
        return

def printToFile(shape,file):
    #Write code to sace 68*2 points on matrix
    print_shape = shape.reshape(1,-1)
    print_shape = print_shape[0]
    for i in print_shape:
        file.write(str(i)+",")
    file.write("\n")

def fun(video_path,fileName,minutesToHold):
    file = open(fileName,"w")
    preProcessedFlag = False
    model = detectors()
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(video_path)
    #check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")
    # Read until video is completed
    start = time.time()
    while cap.isOpened():
        ret,image = cap.read()
        if ret == True:
            #Display Frame
            #cv2.imshow('frame',frame)
            # Converting the image to gray scale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Get faces into webcam's image
            rects = detector(gray, 0)
            # For each detected face, find the landmark.
            for (i, rect) in enumerate(rects):
                # Make the prediction and transfom it to numpy array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                shape1 = np.copy(shape)
                if time.time() - start < minutesToHold*60:
                    printToFile(shape,file)
                else:
                    if preProcessedFlag == False:
                        file.close()
                        preProcessedFlag = True
                        model.getMinMaxScaling(fileName)
                        # file = open(fileName,"w")
                    model.getKSSvalue(shape)                      
                # Draw on our image, all the finded cordinate points (x,y) 
                for (x, y) in shape1:
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            # Show the image
            cv2.imshow("Output", image)
            #Press Q on keyboard to exit
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
fun(0,"abx_test.txt",0.1)