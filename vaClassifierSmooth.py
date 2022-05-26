import numpy as np
import pandas as pd
import cv2
import sklearn
import math
from midiutil.MidiFile import MIDIFile
from sklearn.neural_network import MLPClassifier

inputs = np.empty((0,7))
targets = np.array([])

positions = {
    0: [42, 42], # hi-hat left, hi-hat right
    1: [38, 42], # snare left, hi-hat right
    2: [38, 38], # snare left, snare right
    3: [38, 51], # snare left, ride cymbal right
    4: [38, 43], # snare left, floor tom right
    5: [43, 43], # tom left, tom right
    6: [38, 49] # snare left, crash right
}

def extractFeatures(targetValue, vaFileName):
    global inputs
    global targets

    df = pd.read_csv('./videoAnalysisData/' + vaFileName)
    dfSmooth= df.rolling(100,center=True,win_type='boxcar',min_periods=1).mean()
    dfSmoothNorm =((dfSmooth-dfSmooth.min())/(dfSmooth.max()-dfSmooth.min()))*100

    for index, row in dfSmoothNorm.iterrows():
        rowValues = np.array([row["qom"], row["com_x"], row["com_y"], row["bmi_x"], row["bmi_y"], row["bma_x"], row["bma_y"]])
        inputs = np.append(inputs, rowValues.reshape(1, -1), axis=0)
        targets = np.append(targets, targetValue)

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
        if((i/4) > len(testData.index)):
            break

        time = ((i - startIndex) / 100) * (tempo / 60)
        if hitDetected:
            if(value < sensitivity):
                hitDetected = False
        # If not a double hit, regester a hit
        elif (time - lastHitTime) > 0.1:
            value = y[i]
            if (value > sensitivity) and lastValue > value:
                videoFrame = math.floor(i / 4)
                row = testData.iloc[videoFrame]
                rowValues = np.array([row["qom"], row["com_x"], row["com_y"], row["bmi_x"], row["bmi_y"], row["bma_x"], row["bma_y"]])
                rowValues = rowValues.reshape(1,-1)
                positionPredict = int(mlp.predict(rowValues))
                note = positions[positionPredict][hand]
                lastHitTime = time
                mf.addNote(track, channel, note, time, duration, volume)

            lastValue = value

# PART 1: TRAIN AND TEST SYSTEM

extractFeatures(0, 'hihatLeftHihatRight_data.csv')
extractFeatures(1, 'snareLeftHihatRight_data.csv')
extractFeatures(2, 'snareLeftSnareRight_data.csv')
extractFeatures(3, 'snareLeftRideRight_data.csv')
extractFeatures(4, 'snareLeftTomRight_data.csv')
extractFeatures(5, 'tomLeftTomRight_data.csv')
extractFeatures(6, 'snareLeftCrashRight_data.csv')

# Creating train/test split
feat_train, feat_test, target_train, target_test = sklearn.model_selection.train_test_split(inputs, targets, test_size=0.2)

mlp = MLPClassifier(hidden_layer_sizes=(15,10,5), max_iter=500000, tol=0.0000001, verbose=True)
mlp.fit(feat_train, target_train)
target_predict =  mlp.predict(feat_test)

# Print accuracy score and confusion matrix
print('Number of mislabeled samples %d out of %d' % ((target_test != target_predict).sum(),target_test.size))
print('Accuracy:',sklearn.metrics.accuracy_score(target_test, target_predict))
print(sklearn.metrics.classification_report(target_test, target_predict))
print('confusion matrix')
print(sklearn.metrics.confusion_matrix(target_test,target_predict))

# PART 2: CREATE MIDI FILE BASED ON ML SYSTEM

# load test file
testData = pd.read_csv('./videoAnalysisData/testSlow_data.csv')
testData= testData.rolling(100,center=True,win_type='boxcar',min_periods=1).mean()
testData =((testData-testData.min())/(testData.max()-testData.min()))*100

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
with open("vaClassifierSmooth.mid", 'wb') as outf:
    mf.writeFile(outf)