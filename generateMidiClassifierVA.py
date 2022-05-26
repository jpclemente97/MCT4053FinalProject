from midiutil.MidiFile import MIDIFile
import os
import pandas as pd
import joblib
import math
import cv2
import numpy as np
from sklearn import *
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def writeToMidi(y, hand, sensitivity):
	# create MIDI object with one track
	track = hand
	tempo = 120
	duration = 1
	mf.addTrackName(track, 0, "Test")
	mf.addTempo(track, 0, tempo)
	channel = 0
	volume = 100

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
	print("startindex = " + str(startIndex))
	for i in range(startIndex, y.size):
		if((i/4) > len(testData.index)):
			break

		time = ((i - startIndex) / 100) * (tempo / 60)
		if hitDetected:
			if(value < sensitivity):
				hitDetected = False
		elif (time - lastHitTime) > 0.15:
			value = y[i]
			if (value > sensitivity) and lastValue > value:
				if hand == -1:
					# bass drum
					note = 36
				else:
					videoFrame = math.floor(i / 4)
					row = testData.iloc[videoFrame]
					rowValues = np.array([row["qom"], row["com_x"], row["com_y"], row["bmi_x"], row["bmi_y"], row["bma_x"], row["bma_y"]])
					rowValues = scaler.transform(rowValues.reshape(1,-1))
					rowValues = pca.transform(rowValues)
					positionPredict = int(regressor.predict(rowValues))
					print(positionPredict)
					note = positions[positionPredict][hand]
				#print(time)
				lastHitTime = time
				mf.addNote(track, channel, note, time, duration, volume)

			lastValue = value

positions = {
	-1: [0, 0],
	0: [42, 42], # hi-hat left, hi-hat right
	1: [38, 42], # snare left, hi-hat right
	2: [38, 38], # snare left, snare right
	3: [38, 51], # snare left, ride cymbal right
	4: [38, 43], # snare left, floor tom right
	5: [43, 43], # tom left, tom right
	6: [38, 49], # snare left, crash right
	7: [0, 0],
	8: [0, 0],
	9: [0, 0],
	10: [0, 0]
}

# load regressor
regressor = joblib.load("./joblibFiles/mlp_model_VA.pkl")
scaler = joblib.load("./joblibFiles/scaler_VA.pkl")
pca = joblib.load("./joblibFiles/pca_VA.pkl")

# load video file
video = cv2.VideoCapture('testSlow.mp4')


mf = MIDIFile(2)

df = pd.read_csv('./accelerometerData/leftHandTest1.csv')
leftY = df.loc[:," Accel-Y (g)"].values
df = pd.read_csv ('./accelerometerData/rightHandTest1.csv')
rightY = df.loc[:," Accel-Y (g)"].values
df = pd.read_csv ('./accelerometerData/footTest1.csv')
footY = df.loc[:," Accel-Y (g)"].values

testData = pd.read_csv('./videoAnalysisData/testSlow_data.csv')
testData= testData.rolling(100,center=True,win_type='boxcar',min_periods=1).mean()
testData =((testData-testData.min())/(testData.max()-testData.min()))*100

writeToMidi(leftY, 0, 4)
writeToMidi(rightY, 1, 1.5)
#writeToMidi(footY, -1)
    
# write it to disk
with open("output.mid", 'wb') as outf:
    mf.writeFile(outf)