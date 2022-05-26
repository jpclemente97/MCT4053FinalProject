import numpy as np
import pandas as pd
import cv2
import sklearn
import math
from midiutil.MidiFile import MIDIFile
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

inputs = np.empty((0,2500))
targets = np.empty((inputs.shape[0],2))

instruments = {
    0: 42, # hi-hat
    1: 38, # snare
    2: 51, # ride cymbal
    3: 43, # floor tom
    4: 49 # crash
}

def extractFeatures(fileName, leftHandValue, rightHandValue):
    global inputs
    global targets

    video_capture = cv2.VideoCapture('./videos/' + fileName)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # convert to grayscale, resize, and flatten
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (50, 50))
        frame = frame.flatten()

        inputs = np.append(inputs, frame.reshape(1,-1), axis=0)
        targets = np.append(targets, np.array([[leftHandValue, rightHandValue]]), axis=0) 

def writeToMidi(y, hand, sensitivity):
    global pca
    global scaler
    global video
    global mlp
    global mf

    track = hand
    tempo = 120
    mf.addTrackName(0, 0, "Test")
    mf.addTempo(0, 0, tempo)
    channel = 0
    volume = 100
    # all notes will have a duration of 1
    duration = 1

    # Find start index
    lastValue = 0
    startIndex = 0
    for i,value in enumerate(y):
        if (value > sensitivity) and lastValue > value:
            startIndex = i
            break
        else:
            lastValue = value

    hitDetected = False
    lastHitTime = -1
    for i in range(startIndex, y.size):
        time = ((i - startIndex) / 100) * (tempo / 60)
        if hitDetected:
            if(value < sensitivity):
                hitDetected = False
        # If not a double hit, regester a hit
        elif (time - lastHitTime) > 0.1:
            value = y[i]
            if (value > sensitivity) and lastValue > value:
                videoFrame = math.floor(i / 4)
                video.set(1, videoFrame)
                ret, frame = video.read()
                if ret == False:
                    break

                # convert to grayscale, resize, and flatten
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, (50, 50))
                frame = frame.flatten()
                frame = frame.reshape(1,-1)

                # apply scaler and pca
                frame = scaler.transform(frame)
                frame = pca.transform(frame)

                instrumentPredict = mlp.predict(frame)
                note = instruments[abs(round(instrumentPredict[0][hand]))]
                lastHitTime = time
                mf.addNote(track, channel, note, time, duration, volume)

            lastValue = value

# PART 1: TRAIN AND TEST SYSTEM

extractFeatures('hihatLeftHihatRight.mp4', 0., 0.)
extractFeatures('snareLeftHihatRight.mp4', 1., 0.)
extractFeatures('snareLeftSnareRight.mp4', 1., 1.)
extractFeatures('snareLeftRideRight.mp4', 1., 2.)
extractFeatures('snareLeftTomRight.mp4', 1., 3.)
extractFeatures('tomLeftTomRight.mp4', 3., 3.)
extractFeatures('snareLeftCrashRight.mp4', 1., 4.)

# create test/train split
feat_train, feat_test, target_train, target_test = sklearn.model_selection.train_test_split(inputs, targets, test_size=0.2)

# create and apply scaler
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(feat_train)
feat_train = scaler.transform(feat_train)
feat_test = scaler.transform(feat_test)

# dimensionality reduction
pca = sklearn.decomposition.PCA(n_components=1000)
pca.fit(feat_train)
projected_features_train = pca.transform(feat_train)
projected_features_test = pca.transform(feat_test)

mlp = MLPRegressor(hidden_layer_sizes=(15,10,5), max_iter=500000, tol=0.0000001, verbose=True)
mlp.fit(projected_features_train, target_train)
target_predict =  mlp.predict(projected_features_test)

# Print R2 score
print('r2 score on individual targets',sklearn.metrics.r2_score(target_test, target_predict, multioutput='raw_values'))

# PART 2: CREATE MIDI FILE BASED ON ML SYSTEM

# load test video file
video = cv2.VideoCapture('./videos/testSlow.mp4')

# create MIDI object with two track
mf = MIDIFile(2)

df = pd.read_csv('./accelerometerData/leftHandTest1.csv')
leftY = df.loc[:," Accel-Y (g)"].values
df = pd.read_csv ('./accelerometerData/rightHandTest1.csv')
rightY = df.loc[:," Accel-Y (g)"].values

# the sensitivity values were set via trial and error and can be changed for different data sets
writeToMidi(leftY, 0, 7)
writeToMidi(rightY, 1, 1.5)
    
# write MIDI file to disk
with open("videoRegressor.mid", 'wb') as outf:
    mf.writeFile(outf)