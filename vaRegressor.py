import numpy as np
import pandas as pd
import cv2
import sklearn
import math
from midiutil.MidiFile import MIDIFile
from sklearn.neural_network import MLPRegressor

inputs = np.empty((0,7))
targets = np.empty((0,2))

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

    df = pd.read_csv('./videoAnalysisData/' + fileName)

    for index, row in df.iterrows():
        rowValues = np.array([row["qom"], row["com_x"], row["com_y"], row["bmi_x"], row["bmi_y"], row["bma_x"], row["bma_y"]])
        inputs = np.append(inputs, rowValues.reshape(1, -1), axis=0)
        targets = np.append(targets, np.array([[leftHandValue, rightHandValue]]), axis=0)

def writeToMidi(y, hand, sensitivity):
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
                videoFrame = min(math.floor(i / 4), len(testData.index) - 1)
                row = testData.iloc[videoFrame]
                rowValues = np.array([row["qom"], row["com_x"], row["com_y"], row["bmi_x"], row["bmi_y"], row["bma_x"], row["bma_y"]])
                rowValues = rowValues.reshape(1,-1)
                instrumentPredict = mlp.predict(rowValues)
                if abs(round(instrumentPredict[0][hand])) > 4:
                    instrumentPredict[0][hand] = 0
                note = instruments[abs(round(instrumentPredict[0][hand]))]
                lastHitTime = time
                mf.addNote(track, channel, note, time, duration, volume)

            lastValue = value

# PART 1: TRAIN AND TEST SYSTEM

extractFeatures('hihatLeftHihatRight_data.csv', 0., 0.)
extractFeatures('snareLeftHihatRight_data.csv', 1., 0.)
extractFeatures('snareLeftSnareRight_data.csv', 1., 1.)
extractFeatures('snareLeftRideRight_data.csv', 1., 2.)
extractFeatures('snareLeftTomRight_data.csv', 1., 3.)
extractFeatures('tomLeftTomRight_data.csv', 3., 3.)
extractFeatures('snareLeftCrashRight_data.csv', 1., 4.)

# create test/train split
feat_train, feat_test, target_train, target_test = sklearn.model_selection.train_test_split(inputs, targets, test_size=0.2)


mlp = MLPRegressor(hidden_layer_sizes=(15,10,5), max_iter=500000, tol=0.0000001, verbose=True)
mlp.fit(feat_train, target_train)
target_predict =  mlp.predict(feat_test)

# Print R2 score
print('r2 score on individual targets',sklearn.metrics.r2_score(target_test, target_predict, multioutput='raw_values'))

# PART 2: CREATE MIDI FILE BASED ON ML SYSTEM

# load test file
testData = pd.read_csv('./videoAnalysisData/testSlow_data.csv')

# create MIDI object with one track
mf = MIDIFile(2)

df = pd.read_csv('./accelerometerData/leftHandTest1.csv')
leftY = df.loc[:," Accel-Y (g)"].values
df = pd.read_csv ('./accelerometerData/rightHandTest1.csv')
rightY = df.loc[:," Accel-Y (g)"].values

# the sensitivity values were set via trial and error and can be changed for different data sets
writeToMidi(leftY, 0, 7)
writeToMidi(rightY, 1, 1.5)
    
# write MIDI file to disk
with open("vaRegressor.mid", 'wb') as outf:
    mf.writeFile(outf)

