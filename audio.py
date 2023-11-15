import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from math import log2
from scipy.fftpack import fft


class Audio():
    def __init__(self, audio=None, sampleRate=None, file=None):
        if audio is None and sampleRate is None:
            self._frequency, self._samples = sf.read(file)
            self.readFile = True
        if file is None:
            self.audio = audio
            self.sampleRate = sampleRate
            self.readFile = False

    def frequency(self):
        if self.readFile:
            return self._frequency
        else:
            return self.sampleRate

    def samples(self):
        if self.readFile:
            return self._samples
        else:
            return self.audio
    
    def dft(self):
        return fft(self.samples())
    
    def M(self, N, l, referenceFreq, sampleFreq):
        if l == 0:
            return -1
        return round(12 * log2((sampleFreq * l)/(N * referenceFreq))) % 12
        
    
    def calculatePCP(self, audioSignal):
        sampleFrequency = self.frequency()
        referenceFrequency = 130.81

        N = len(audioSignal)

        pcp = [0 for _ in range(12)]

        for p in range(12):
            for l in range(N//2):
                if p == self.M(N, l, referenceFrequency, sampleFrequency):
                    pcp[p] += abs(audioSignal[l]) ** 2

        # Normalizing PCP
        pcpNormalized = [p / sum(pcp) for p in pcp]

        return pcpNormalized
    
    def getPCP(self):
        audioSignal = self.dft()
        self.audioPCP = self.calculatePCP(audioSignal)
        return self.audioPCP
    
    def plotPCP(self, title=None, ax=None):
        semitones = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
        yPositions = np.arange(len(semitones))
        profiles = self.audioPCP

        if ax is None:
            fig, ax = plt.subplots()

        ax.bar(yPositions, profiles, align='center', alpha=0.5)
        ax.set_xticks(yPositions)
        ax.set_xticklabels(semitones)
        ax.set_ylabel('Energy')

        # plt.bar(yPositions, profiles, align='center', alpha=0.5)
        # plt.xticks(yPositions, semitones)
        # plt.ylabel('Energy')
        if title is None:
            ax.set_title('PCP')
        else:
            ax.set_title(title)




        
    

