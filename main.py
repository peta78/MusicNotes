import array
import numpy as np
import math
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.utils import get_array_type

sound = AudioSegment.from_file(file='hunter.wav')
left = sound.split_to_mono()[0]

bit_depth = left.sample_width * 8
array_type = get_array_type(bit_depth)

l = np.array(array.array(array_type, left._data))

print(l.shape, bit_depth)
print(sound)
print(sound.frame_rate, sound.duration_seconds, l.shape[0] / sound.frame_rate)

# https://pages.mtu.edu/~suits/notefreqs.html
notes = [
    'C6', 'C#6', 'D6', 'D#6', 'E6', 'F6', 'F#6', 'G6', 'G#6', 'A6', 'A#6', 'B6',
    'C7', 'C#7', 'D7', 'D#7', 'E7', 'F7', 'F#7', 'G7', 'G#7', 'A7', 'A#7', 'B7']
noteFrequencies = [
    1046.5, 1108.73, 1174.66, 1244.51, 1318.51, 1396.91, 1479.98, 1567.98, 1661.22, 1760, 1864.66, 1975.53,
    2093.005, 2217.461, 2349.318, 2489.016, 2637.021, 2793.826, 2959.956, 3135.964, 3322.438, 3520.000, 3729.310, 3951.066,
    ]

closestFreq = [float('inf') for i in range(len(notes))]
closestInd = [-1 for i in range(len(notes))]

fftsize = 10000
for i in range(1,fftsize//2):
    fr = 1.0/ (fftsize / i) * sound.frame_rate
    for j in range(len(notes)):
        if abs(closestFreq[j]-noteFrequencies[j]) > abs(fr-noteFrequencies[j]):
            closestFreq[j] = fr
            closestInd[j] = i

for i in range(len(notes)):
    print(notes[i], noteFrequencies[i], closestFreq[i], closestInd[i])

res = []
resX = []
t = []
mv = 0
for i in range(l.shape[0]-fftsize):
    t.append(i / sound.frame_rate)
    ffta = np.abs(np.fft.fft(l[i:i+fftsize])[0:fftsize//2])
    tmp = []
    tmpX = []
    mv = max(mv,max(ffta[:-4]+ffta[1:-3]+ffta[2:-2]+ffta[3:-1]+ffta[4:])/5.0)
    tmpX.append(np.sum(ffta[:closestInd[0]-2])/(closestInd[0]-2))
    for j in range(len(notes)):
        tmp.append(np.sum(ffta[closestInd[j]-2:closestInd[j]+3])/5.0)
        if j != len(notes)-1:
            tmpX.append(np.sum(ffta[closestInd[j]+3:closestInd[j+1]-2])/(closestInd[j+1]-2 - (closestInd[j]+3)))
    tmpX.append(np.sum(ffta[closestInd[-1]+3:])/(len(ffta) - (closestInd[-1]+3)))
    res.append(tmp)
    resX.append(tmpX)

res = np.array(res)
resX = np.array(resX)
print(res.shape)

print(mv)

fig, axs = plt.subplots(1, 2)
for i in range(len(notes)):
    axs[0].plot(t, res[:,i]/mv, label=notes[i])
    axs[1].plot(t, 2.0 * i + 1.0 + res[:,i]/mv, 'b')
    axs[1].text(0, 2.0 * i + 1.0 + 0.25, notes[i])
    axs[1].plot(t, 1.0 + 2.0 * i * np.ones(len(t)), 'b:')
for i in range(resX.shape[1]):
    axs[0].plot(t, resX[:,i]/mv, 'r:', alpha=0.25)
    axs[1].plot(t, 2.0 * i + resX[:,i]/mv, 'r')
    axs[1].text(0, 2.0 * i + 0.25, 'X')
    axs[1].plot(t, 2.0 * i * np.ones(len(t)), 'r:')
axs[0].set_ylim(0,1)
axs[0].legend(loc='upper left')
plt.show()
