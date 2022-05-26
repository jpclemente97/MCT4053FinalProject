import numpy as np
import pandas as pd
import cv2
import sklearn
import math
from midiutil.MidiFile import MIDIFile
from sklearn.neural_network import MLPRegressor

inputs = np.empty((0,13))
targets = np.empty((0,2))

instruments = {
    0: 42, # hi-hat
    1: 38, # snare
    2: 51, # ride cymbal
    3: 43, # floor tom
    4: 49 # crash
}

def extractFeatures(vaFileName, leftFileName, rightFileName, leftValue, rightValue):
    global inputs
    global targets

    df = pd.read_csv('./videoAnalysisData/' + vaFileName)
    dfSmooth= df.rolling(100,center=True,win_type='boxcar',min_periods=1).mean()
    dfSmoothNorm =((dfSmooth-dfSmooth.min())/(dfSmooth.max()-dfSmooth.min()))*100

    dfLeft = pd.read_csv('./accelerometerData/' + leftFileName)
    dfLeftSmooth= dfLeft.rolling(4,center=True,win_type='boxcar',min_periods=1).mean()
    dfLeftSmoothNorm =((dfLeftSmooth-dfLeftSmooth.min())/(dfLeftSmooth.max()-dfLeftSmooth.min()))*100

    dfRight = pd.read_csv('./accelerometerData/' + rightFileName)
    dfRightSmooth= dfRight.rolling(4,center=True,win_type='boxcar',min_periods=1).mean()
    dfRightSmoothNorm =((dfRightSmooth-dfRightSmooth.min())/(dfRightSmooth.max()-dfRightSmooth.min()))*100

    for index, row in dfSmoothNorm.iterrows():
        vaRowValues = np.array([row["qom"], row["com_x"], row["com_y"], row["bmi_x"], row["bmi_y"], row["bma_x"], row["bma_y"]])

        accelerometerRowIndex = min(len(dfRight.index) - 1, (index + 1) * 4)
        leftRow = dfLeftSmoothNorm.iloc[accelerometerRowIndex]
        rightRow = dfRightSmoothNorm.iloc[accelerometerRowIndex]
        accelerometerRowValues = np.array([leftRow["x"], leftRow["y"], leftRow["z"], rightRow["x"], rightRow["y"], rightRow["z"]])

        rowValues = np.concatenate((vaRowValues, accelerometerRowValues))
        inputs = np.append(inputs, rowValues.reshape(1, -1), axis=0)
        targets = np.append(targets, np.array([[leftValue, rightValue]]), axis=0)

def writeToMidi(y, dfLeftSmoothNorm, dfRightSmoothNorm, hand, sensitivity):
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
                videoFrame = min(math.floor(i / 4), len(testData.index) - 1)
                row = testData.iloc[videoFrame]
                rowValues = np.array([row["qom"], row["com_x"], row["com_y"], row["bmi_x"], row["bmi_y"], row["bma_x"], row["bma_y"]])
                
                leftRow = dfLeftSmoothNorm.iloc[i]
                rightRow = dfRightSmoothNorm.iloc[i]
                accelerometerRowValues = np.array([leftRow['Accel-X (g)'], leftRow[' Accel-Y (g)'], leftRow[' Accel-Z (g)'], rightRow['Accel-X (g)'], rightRow[' Accel-Y (g)'], rightRow[' Accel-Z (g)']])

                rowValues = np.concatenate((rowValues, accelerometerRowValues))
                rowValues = rowValues.reshape(1,-1)
                instrumentPredict = mlp.predict(rowValues)
                if abs(round(instrumentPredict[0][hand])) > 4:
                    instrumentPredict[0][hand] = 0
                note = instruments[abs(round(instrumentPredict[0][hand]))]
                lastHitTime = time
                mf.addNote(track, channel, note, time, duration, volume)

            lastValue = value

# PART 1: TRAIN AND TEST SYSTEM

extractFeatures('hihatLeftHihatRight_data.csv', 'leftHandBothHihat.csv', 'rightHandBothHihat.csv', 0., 0.)
extractFeatures('snareLeftHihatRight_data.csv', 'leftHandSnareAndHihat.csv', 'rightHandSnareAndHihat.csv',  1., 0.)
extractFeatures('snareLeftSnareRight_data.csv', 'leftHandBothSnare.csv', 'rightHandBothSnare.csv', 1., 1.)
extractFeatures('snareLeftRideRight_data.csv', 'leftHandRideAndSnare.csv', 'rightHandRideAndSnare.csv', 1., 2.)
extractFeatures('snareLeftTomRight_data.csv', 'leftHandTomAndSnare.csv', 'rightHandTomAndSnare.csv', 1., 3.)
extractFeatures('tomLeftTomRight_data.csv', 'leftHandBothTom.csv', 'rightHandBothTom.csv', 3., 3.)
extractFeatures('snareLeftCrashRight_data.csv', 'leftHandCrashAndSnare.csv', 'rightHandCrashAndSnare.csv', 1., 4.)

# Creating train/test split
feat_train, feat_test, target_train, target_test = sklearn.model_selection.train_test_split(inputs, targets, test_size=0.2)

mlp = MLPRegressor(hidden_layer_sizes=(15,10,5), max_iter=500000, tol=0.0000001, verbose=True)
mlp.fit(feat_train, target_train)
target_predict =  mlp.predict(feat_test)

# Print R2 score
print('r2 score on individual targets',sklearn.metrics.r2_score(target_test, target_predict, multioutput='raw_values'))

# PART 2: CREATE MIDI FILE BASED ON ML SYSTEM

# load test file
testData = pd.read_csv('./videoAnalysisData/testSlow_data.csv')
testData= testData.rolling(100,center=True,win_type='boxcar',min_periods=1).mean()
testData =((testData-testData.min())/(testData.max()-testData.min()))*100

# create MIDI object with one track
mf = MIDIFile(2)

dfLeft = pd.read_csv('./accelerometerData/leftHandTest1.csv')
leftY = dfLeft.loc[:," Accel-Y (g)"].values
dfLeftSmooth= dfLeft.rolling(4,center=True,win_type='boxcar',min_periods=1).mean()
dfLeftSmoothNorm =((dfLeftSmooth-dfLeftSmooth.min())/(dfLeftSmooth.max()-dfLeftSmooth.min()))*100

dfRight = pd.read_csv('./accelerometerData/rightHandTest1.csv')
rightY = dfRight.loc[:," Accel-Y (g)"].values
dfRightSmooth= dfRight.rolling(4,center=True,win_type='boxcar',min_periods=1).mean()
dfRightSmoothNorm =((dfRightSmooth-dfRightSmooth.min())/(dfRightSmooth.max()-dfRightSmooth.min()))*100

# the sensitivity values were set via trial and error and can be changed for different data sets
writeToMidi(leftY, dfLeftSmoothNorm, dfRightSmoothNorm, 0, 7)
writeToMidi(rightY, dfLeftSmoothNorm, dfRightSmoothNorm, 1, 1.5)
    
# write MIDI file to disk
with open("vaRegressorAccelerometerSmooth.mid", 'wb') as outf:
    mf.writeFile(outf)