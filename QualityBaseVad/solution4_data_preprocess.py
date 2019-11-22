import glob
import os

import pandas as pd
from scipy.io import wavfile as wf
from tqdm import tqdm

from solution4_utils import *


def get_one_audio_break_feature(signal, sr, sample_index, mfccs):
    frame_start_index = sample_index
    frame_end_index = sample_index + win_length * sr
    if frame_end_index >= len(signal):
        frame_end_index = len(signal) - 1
    before_sample_index = int(sample_index - 0.01 * sr)
    after_sample_index = int(sample_index + 0.01 * sr)
    if before_sample_index < 0:
        before_sample_index = 0
    if after_sample_index >= len(signal):
        after_sample_index = len(signal) - 1
    hop_length_sample = int(hop_length * sr)
    frame_index = sample_index // hop_length_sample
    before_frame_index = before_sample_index // hop_length_sample
    after_frame_index = after_sample_index // hop_length_sample
    frame_power_power = power_ratio(signal, sample_index, sr)
    frame_local_power = local_power(signal, frame_start_index, frame_end_index)
    sample_mfcc = mfccs[:, frame_index].flatten()
    before_mfcc = mfccs[:, before_frame_index].flatten()
    after_mfcc = mfccs[:, after_frame_index].flatten()

    return frame_power_power, frame_local_power, sample_mfcc, before_mfcc, after_mfcc


def get_audio_data(wav_path, break_sample_indexes, labels):
    sr, audio = wf.read(wav_path)
    wav_basename = os.path.basename(wav_path)[:-4]
    mfccs = get_mfcc(audio, sr=sr)
    audio_feature = []
    for break_sample_index, label in zip(break_sample_indexes, labels):
        break_feature = []
        frame_power_power, frame_local_power, sample_mfcc, before_mfcc, after_mfcc = get_one_audio_break_feature(audio,
                                                                                                                 sr,
                                                                                                                 break_sample_index,
                                                                                                                 mfccs)
        break_feature.append('{}_{}'.format(wav_basename, break_sample_index))
        break_feature.append(frame_power_power)
        break_feature.append(frame_local_power)
        break_feature.extend(sample_mfcc)
        break_feature.extend(before_mfcc)
        break_feature.extend(after_mfcc)
        break_feature.append(int(label))
        audio_feature.append(break_feature)

    return np.asarray(audio_feature)


def make_dataset(wav_dir, metadata_path, dataset_path):
    wav_paths = glob.glob(os.path.join(wav_dir, '*.wav'))
    metadata = pd.read_csv(metadata_path)
    wavs_features = None
    for wav_path in tqdm(wav_paths):
        wav_basename = os.path.basename(wav_path)[:-4]
        wav_metadata = metadata[metadata['wav_filename'] == wav_basename]
        break_sample_indexes = wav_metadata['index'].values
        labels = wav_metadata['label'].values.astype('int')
        audio_feature = get_audio_data(wav_path, break_sample_indexes, labels)
        if len(audio_feature) < 1:
            continue
        wavs_features = audio_feature if wavs_features is None else np.concatenate((wavs_features, audio_feature))

    print('dataset shape: ', wavs_features.shape)
    np.save(dataset_path, wavs_features)


if __name__ == '__main__':
    root_dir = 'data'
    wav_dir = os.path.join(root_dir, 'resample_dataset')
    metadata_path = os.path.join(root_dir, 'solution4_metadata2.csv')
    dataset_path = os.path.join(root_dir, 'solution4_data.npy')
    make_dataset(wav_dir, metadata_path, dataset_path)
