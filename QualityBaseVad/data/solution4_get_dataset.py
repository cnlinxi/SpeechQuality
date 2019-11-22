import glob
import os

import librosa
import scipy.io.wavfile as wf
from pydub import AudioSegment
from tqdm import tqdm

from solution4_base_vad import VAD


def normalize_audio(input_dir='data/resample_dataset', output_dir='normalized_dataset', target_dbfs=-26):
    wav_paths = glob.glob(os.path.join(input_dir, '*.wav'))
    os.makedirs(output_dir, exist_ok=True)
    for wav_path in tqdm(wav_paths):
        audio = AudioSegment.from_file(wav_path, format=wav_path[-3:])
        audio.remove_dc_offset()
        audio.apply_gain(target_dbfs - audio.dBFS)
        audio.export(os.path.join(output_dir, os.path.basename(wav_path)), format=wav_path[-3:])


def resample_audio(input_dir='data/my_dataset', output_dir='data/resample_dataset'):
    wav_paths = glob.glob(os.path.join(input_dir, '*.wav'))
    durations = 0.
    for wav_path in tqdm(wav_paths):
        wav_basename = os.path.basename(wav_path)
        y, sr = librosa.load(wav_path, sr=8000)
        durations += float(len(y)) / sr
        wf.write(os.path.join(output_dir, wav_basename), rate=8000, data=y)
    print('dataset duration: {} s'.format(durations))


def mark_speech_end_into_csv(input_dir='data/resample_dataset', dataset_path='solution4_metadata5.csv'):
    wav_paths = [os.path.join(input_dir, x) for x in os.listdir(input_dir)]
    with open(dataset_path, 'wb') as fin:
        fin.write('audio_down_id,wav_filename,index,time,label,unnature,sharpdecline,pairbreak\n'.encode('utf-8'))
        for wav_path in tqdm(wav_paths):
            sr, signal = wf.read(wav_path)
            wav_basename = os.path.basename(wav_path)[:-4]
            vad = VAD(signal, sr, nFFT=512, win_length=0.025, hop_length=0.01, theshold=0.99)
            is_speech = False
            speech_infos = []
            speech_info = None
            for index, ele in enumerate(vad):
                if ele == 1 and (not is_speech):  # speech start
                    speech_info = {'start_time': index / sr}
                    is_speech = True
                elif ele == 0 and is_speech:  # speech end
                    speech_info['end_time'] = index / sr
                    speech_infos.append(speech_info)
                    is_speech = False
                    fin.write('{}_{},{},{},{}\n'.format(wav_basename,
                                                        index,
                                                        wav_basename,
                                                        index,
                                                        index / sr).encode('utf-8'))


if __name__ == '__main__':
    normalize_audio()
