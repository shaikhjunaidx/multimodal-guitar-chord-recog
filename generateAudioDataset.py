import os
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import soundfile as sf  # for reading and writing audio files
import numpy as np  # for adding noise
from audio import Audio
from pedalboard import Reverb


sampleRate = 44100
plotPCP = False
saveAugmented = False
augmentData = True

# Directory containing the .wav files
dataDir = "multimodal-guitar-chord-recog/chord_audio_files"

# Function to apply reverb effect to an audio file
def applyReverb(audio, sampleRate, audioPath):
    reverb = Reverb(room_size=0.75)
    reverbAudio = reverb(audio, sampleRate)

    if saveAugmented:
        outputDir = "multimodal-guitar-chord-recog/augmented_chord_audio" + "/reverbed"
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

        outputFilePath =  outputDir + "/reverbed_" + os.path.basename(audioPath)
        sf.write(outputFilePath, reverbAudio, sampleRate)

    return Audio(audio=reverbAudio, sampleRate=sampleRate)


# Function to add noise to an audio file
def addNoise(audio, sampleRate, audioPath, noiseFactor=0.05):
    noise = np.random.normal(0, audio.std(), audio.size)
    noisyAudio = audio + noise * noiseFactor

    if saveAugmented:
        outputDir = "multimodal-guitar-chord-recog/augmented_chord_audio" + "/noisy"
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

        outputFilePath = outputDir + "/noisy_" + os.path.basename(audioPath)
        sf.write(outputFilePath, noisyAudio, sampleRate)

    return Audio(audio=noisyAudio, sampleRate=sampleRate)


def timeStretch(audio, sampleRate, audioPath, stretchRate=0.7):
    stretchedAudio = librosa.effects.time_stretch(audio, rate=stretchRate)

    if saveAugmented:
        outputDir = "multimodal-guitar-chord-recog/augmented_chord_audio" + "/stretched"
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

        outputFilePath = outputDir + "/stretched" + os.path.basename(audioPath)
        sf.write(outputFilePath, stretchedAudio, sampleRate)

    return Audio(audio=stretchedAudio, sampleRate=sampleRate)


# Function to read .wav files and extract features
def processWavFiles(directory):

    chordData = []  # List to store the processed data

    # Iterate through subdirectories (chord labels)
    for chordLabel in os.listdir(directory):
        print("Processing chord " + chordLabel)
        chordDir = os.path.join(directory, chordLabel)

        # Check if the item in the directory is a subdirectory (chord label)
        if os.path.isdir(chordDir):

            # Iterate through .wav files in the subdirectory
            for file in os.listdir(chordDir):
                if file.endswith(".wav"):
                    print("\t" + file)

                    # Initialize chord vectors for normal, reverb, and noise versions
                    chordVectors = []

                    # Process the normal audio file
                    filePath = os.path.join(chordDir, file)
                    audio, sampleRate = sf.read(filePath)

                    normalAudio = Audio(audio=audio, sampleRate=sampleRate)
                    normalProfile = normalAudio.getPCP()
                    chordVectors.append(normalProfile)  # Append pitch class profiles

                    if augmentData:
                        # Apply reverb effect to the audio file
                        reverbAudio = applyReverb(audio, sampleRate, filePath)
                        reverbProfile = reverbAudio.getPCP()
                        chordVectors.append(reverbProfile)

                        # Add noise to the audio file
                        noisyAudio = addNoise(audio, sampleRate, filePath, noiseFactor=0.3)
                        noisyProfile = noisyAudio.getPCP()
                        chordVectors.append(noisyProfile)
                        
                        # Stretch the audio file
                        stretchedAudio = timeStretch(audio, sampleRate, filePath)
                        stretchedProfile = stretchedAudio.getPCP()
                        chordVectors.append(stretchedProfile)

                        if plotPCP:
                            # Create subplots
                            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                            fig.suptitle('PCP Comparison for chord ' + chordLabel.upper(), fontsize=16)
                            normalAudio.plotPCP('Normal PCP', ax=axes[0, 0])
                            reverbAudio.plotPCP('Reverb PCP', ax=axes[0, 1])
                            noisyAudio.plotPCP('Noisy PCP', ax=axes[1, 0])
                            stretchedAudio.plotPCP('Stretched PCP', ax=axes[1, 1])

                            plt.tight_layout()
                            plt.subplots_adjust(top=0.85)

                            plt.show()
                    else:
                        if plotPCP:
                            fig, axes = plt.subplots(1, 1, figsize=(5, 4))
                            fig.suptitle('PCP Comparison', fontsize=16)
                            normalAudio.plotPCP('Normal PCP', ax=axes)
                            plt.tight_layout()
                            plt.subplots_adjust(top=0.5)

                            plt.show()

                    # Add chord label to each version
                    for chordVector in chordVectors:
                        chordVector.append(chordLabel.upper())  # Append chord label
                        chordData.append(chordVector)
                    
    return chordData


# Process .wav files and create a dataset
dataset = processWavFiles(dataDir)

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(dataset)

# Define column names for the DataFrame (12 pitch class profiles + 1 label)
columnNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] + ['ChordLabel']

# Set column names
df.columns = columnNames

# Save the DataFrame to a CSV file
csvFilePath = "multimodal-guitar-chord-recog/audioPCP.csv" 
df.to_csv(csvFilePath, index=False)  # Save to CSV without row indices

print(f"Data saved to {csvFilePath}")