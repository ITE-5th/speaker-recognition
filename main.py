import shutil
from subprocess import call

from python_speech_features.base import mfcc
import numpy as np
import scipy.io.wavfile as wav
import os
import glob
from sklearn.mixture import GaussianMixture
import pickle


def train_gmms(directory):
    gmms = dict()
    components = 32
    directory += "/TRAIN/"
    for d in range(1, 9):
        current_dir = directory + "DR{}/".format(d)
        speakers = os.listdir(current_dir)
        for speaker in speakers:
            speaker_dir = current_dir + speaker + "/"
            print(speaker_dir)
            gmm = GaussianMixture(components)
            voices = glob.glob(speaker_dir + "*.WAV")
            # just to test the system
            voices = [v for v in voices if "SA1" not in v]
            mfccs = []
            for voice in voices:
                temp = speaker_dir + "temp.wav"
                raw = speaker_dir + "t.rawaudio"
                shutil.move(voice, temp)
                call(['sox', temp, voice])
                shutil.move(temp, raw)
                (rate, sig) = wav.read(voice)
                mfcc_feat = mfcc(sig, rate)
                mfccs.append(mfcc_feat)
            mfccs = np.vstack(mfccs)
            gmm.fit(mfccs)
            gmms[speaker] = gmm
    return gmms


def test_gmms(gmms, voice_path):
    max_score = -1000000
    target_speaker = None
    (rate, sig) = wav.read(voice_path)
    mfcc_feat = mfcc(sig, rate)
    for speaker, gmm in gmms.items():
        scores = gmm.score_samples(mfcc_feat)
        score = scores.mean()
        if score > max_score:
            max_score = score
            target_speaker = speaker
    return target_speaker


def store_gmms(directory):
    gmms = train_gmms(directory)
    with open('gmms.pkl', 'wb') as f:
        pickle.dump(gmms, f)


def load_gmms():
    with open("gmms.pkl", "rb") as f:
        return pickle.load(f)


if __name__ == '__main__':
    store_gmms("./data")
    # gmms = load_gmms()
    # voice_path = "./data/TRAIN/DR8/MMWS0/SA1.WAV"
    # speaker = test_gmms(gmms, voice_path)
    # print(speaker)
