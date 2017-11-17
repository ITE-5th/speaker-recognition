import shutil
from subprocess import call

from python_speech_features.base import mfcc
import numpy as np
import scipy.io.wavfile as wav
import os
import glob
from sklearn.mixture import GaussianMixture
import pickle
from multiprocessing import Pool, cpu_count


def train_dir_timit(param):
    directory, dir_number = param
    return train_people(directory + "DR{}".format(dir_number))


def train_person(person_dir):
    print(person_dir)
    components = 32
    gmm = GaussianMixture(components)
    person_dir += "/"
    voices = glob.glob(person_dir + "*.WAV")
    mfccs = []
    for voice in voices:
        temp = person_dir + "temp.wav"
        raw = person_dir + "t.rawaudio"
        shutil.move(voice, temp)
        call(['sox', temp, voice])
        shutil.move(temp, raw)
        (rate, sig) = wav.read(voice)
        mfcc_feat = mfcc(sig, rate)
        mfccs.append(mfcc_feat)
        # os.system("rm -f {}".format(raw))
        # os.system("rm -f {}".format(temp))
    mfccs = np.vstack(mfccs)
    gmm.fit(mfccs)
    return gmm


def train_people(people_dir):
    names = os.listdir(people_dir)
    gmms = {name: train_person(people_dir + "/" + name) for name in names}
    return gmms


def train_gmms_from_timit(directory):
    directory += "/TRAIN/"
    with Pool(cpu_count()) as p:
        result = p.map(train_dir_timit, [(directory, i) for i in range(1, 9)])
    gmms = dict()
    for res in result:
        gmms.update(res)
    return gmms


def detect(gmms, voice_paths):
    result = []
    for voice_path in voice_paths:
        max_score = -1e6
        target_speaker = None
        (rate, sig) = wav.read(voice_path)
        mfcc_feat = mfcc(sig, rate)
        for speaker, gmm in gmms.items():
            scores = gmm.score_samples(mfcc_feat)
            score = scores.mean()
            print("{} = {}".format(speaker, score))
            if score > max_score:
                max_score = score
                target_speaker = speaker
        result.append(target_speaker)
    return result


def store_gmms(directory, model_path):
    gmms = train_gmms_from_timit(directory)
    with open(model_path, 'wb') as f:
        pickle.dump(gmms, f)


def load_gmms(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


if __name__ == '__main__':
    model_path = "gmms.pkl"
    store_gmms("./data", model_path)
    # uncomment the following lines after training
    # gmms = load_gmms(model_path)
    # voice_to_test_paths = ["data/jackson.wav", "data/king.wav", "data/devil.wav"]
    # speakers = detect(gmms, voice_to_test_paths)
    # print(speakers)
