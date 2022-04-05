from midiutil.MidiFile import MIDIFile
import os
import pandas as pd

# create MIDI object with one trackc
mf = MIDIFile(1)
track = 0

tempo = 120

mf.addTrackName(0, 0, "Test")
mf.addTempo(0, 0, tempo)

channel = 0
volume = 100

df = pd.read_csv ('drum.csv')
print(df.keys())
y = df.loc[:," Accel-Y (g)"].values

# Find start index
lastValue = 0
startIndex = 0
for i,value in enumerate(y):
	if (value > 3.25) and lastValue > value:
		startIndex = i
		break
	else:
		lastValue = value

print("startindex = " + str(startIndex))
for i in range(startIndex, y.size):
	value = y[i]
	if (value > 3.25) and lastValue > value:
		print(i)
		pitch = 60
		time = ((i - startIndex) / 100) * (tempo / 60)
		duration = 1
		mf.addNote(track, channel, pitch, time, duration, volume)

	lastValue = value
    

# write it to disk
with open("output.mid", 'wb') as outf:
    mf.writeFile(outf)