import glob
import math
import pickle
import random

import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture


def remove_silence(fs,
                   signal,
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
    return retsig.astype(orig_dtype), fs


class Speaker:

    def __init__(self, name, components=32):
        self.name = name
        self.gmm = GaussianMixture(n_components=components, max_iter=200, n_init=2)

    def train_from_directory(self, directory):
        if directory[-1] != "/":
            directory += "/"
        voices = glob.glob(directory + "*.wav")
        random.shuffle(voices)
        feats = []
        for i in range(len(voices)):
            voice = voices[i]
            feat = Speaker.calculate_mfcc(voice)
            feats.append(feat)
        feats = np.vstack(feats)
        self.gmm.fit(feats)

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
        # not sure?
        audio, rate = remove_silence(rate, audio)
        mfcc_feat = mfcc(audio, rate, numcep=20)
        mfcc_feat = preprocessing.scale(mfcc_feat)
        delta = Speaker.calculate_delta(mfcc_feat)
        combined = np.hstack((mfcc_feat, delta))
        return combined

    def predict(self, mfccs):
        return self.gmm.score(mfccs)

    def save(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)


class SpeakersModel:

    def __init__(self, speakers):
        self.speakers = speakers

    def verify_speaker(self, voice_path, claimed_speaker, threshold=0.5):
        speakers = self.speakers
        try:
            mfccs = Speaker.calculate_mfcc(voice_path)
            claimed_speaker = [x for x in speakers if x.name == claimed_speaker][0]
            other_speakers = [x for x in speakers if x.name != claimed_speaker]
            claimed_speaker_score = claimed_speaker.predict(mfccs)
            t = [x.predict(mfccs) for x in other_speakers]
            t = np.array(t)
            other_speakers_score = np.exp(t).sum()
            result = math.exp(claimed_speaker_score) / other_speakers_score
            print(result)
            return result >= threshold
        except Exception:
            raise SpeakerNotFoundException()

    def predict_speaker(self, voice_path):
        speakers = self.speakers
        max_score = -1e9
        max_speaker = None
        mfccs = Speaker.calculate_mfcc(voice_path)
        for speaker in speakers:
            score = speaker.predict(mfccs)
            if score > max_score:
                max_score = score
                max_speaker = speaker
        return max_speaker

    def save(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)


class SpeakerNotFoundException(Exception):
    pass


if __name__ == '__main__':

    # training

    # speakers = []
    # paths = sorted(os.listdir("data"), key=lambda x: x.lower())
    # for path in paths:
    #     temp = f"data/{path}/wav"
    #     name = path[:path.rfind("-")].title()
    #     print(name)
    #     speaker = Speaker(name)
    #     speaker.train_from_directory(temp)
    #     speakers.append(speaker)
    # model = SpeakersModel(speakers)
    # model.save("models/gmms.model")

    # testing
    model: SpeakersModel = SpeakersModel.load("models/gmms.model")
    print(model.verify_speaker("b0225.wav", "Arjuan-20100820"))
