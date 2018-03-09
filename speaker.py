import glob
import pickle
import random

import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture


def remove_silence(fs, signal,
                   frame_duration=0.02,
                   frame_shift=0.01,
                   perc=0.15):
    orig_dtype = type(signal[0])
    typeinfo = np.iinfo(orig_dtype)
    is_unsigned = typeinfo.min >= 0
    signal = signal.astype(np.int64)
    if is_unsigned:
        signal = signal - (typeinfo.max + 1) / 2
    siglen = len(signal)
    retsig = np.zeros(siglen, dtype=np.int64)
    frame_length = int(frame_duration * fs)
    frame_shift_length = int(frame_shift * fs)
    new_siglen = 0
    i = 0
    average_energy = np.sum(signal ** 2) / float(siglen)
    while i < siglen:
        subsig = signal[i:i + frame_length]
        ave_energy = np.sum(subsig ** 2) / float(len(subsig))
        if ave_energy < average_energy * perc:
            i += frame_length
        else:
            sigaddlen = min(frame_shift_length, len(subsig))
            retsig[new_siglen:new_siglen + sigaddlen] = subsig[:sigaddlen]
            new_siglen += sigaddlen
            i += frame_shift_length
    retsig = retsig[:new_siglen]
    if is_unsigned:
        retsig = retsig + typeinfo.max / 2
    return retsig.astype(orig_dtype)


class Speaker:

    def __init__(self, name, components=32, threshold=0.5, train_size=0.9):
        self.name = name
        self.threshold = threshold
        self.train_size = train_size
        self.gmm = GaussianMixture(n_components=components, max_iter=200, n_init=5)

    def train_from_directory(self, directory):
        if directory[-1] != "/":
            directory += "/"
        voices = glob.glob(directory + "*.wav")
        random.shuffle(voices)
        temp = int(self.train_size * len(voices))
        feats = []
        for i in range(temp):
            voice = voices[i]
            feat = self.calculate_mfcc(voice)
            feats.append(feat)
        feats = np.vstack(feats)
        self.gmm.fit(feats)
        feats = []
        for i in range(temp, len(voices)):
            voice = voices[i]
            feat = self.calculate_mfcc(voice)
            feats.append(feat)
        feats = np.vstack(feats)
        # densities
        t = self.gmm.score_samples(feats)
        # back to props, but they have no meaning!, if we integrate over 40 dimensions(mfcc + delta) we will get 1 :).
        t = np.exp(t)
        # of course, they are very small!
        print(t)

    @staticmethod
    def calculate_mfcc(voice):
        (rate, sig) = wav.read(voice)
        return Speaker.extract_features(sig, rate)

    @staticmethod
    def calculate_delta(array):
        rows, cols = array.shape
        deltas = np.zeros((rows, 20))
        n = 2
        for i in range(rows):
            index = []
            j = 1
            while j <= n:
                if i - j < 0:
                    first = 0
                else:
                    first = i - j
                if i + j > rows - 1:
                    second = rows - 1
                else:
                    second = i + j
                index.append((second, first))
                j += 1
            deltas[i] = (array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]] - array[index[1][1]]))) / 10
        return deltas

    @staticmethod
    def extract_features(audio, rate):
        mfcc_feat = mfcc(audio, rate, numcep=20)
        mfcc_feat = preprocessing.scale(mfcc_feat)
        delta = Speaker.calculate_delta(mfcc_feat)
        combined = np.hstack((mfcc_feat, delta))
        return combined

    def save(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, model_path):
        with open(model_path, 'rb') as f:
            res = pickle.load(f)
            self.gmm = res.gmm
            self.name = res.name
            self.threshold = res.threshold
            self.train_size = res.train_size


if __name__ == '__main__':
    name = "Anthony"
    speaker = Speaker(name)
    speaker.train_from_directory("data/anthonyschaller-20071221-/wav")
    speaker.save(f"models/{name}.model")
