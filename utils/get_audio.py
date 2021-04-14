import os, glob, random

import librosa
import numpy as np


class GetAudio():
    def __init__(self, data_path, data_format="*/*.flac"):
        self.data_path = data_path
        self.data_format = data_format
        _, self.id_names, _ = next(os.walk(self.data_path))
        self.sampling_rate = 16000
        self.wave_len = 48000
        self.n_fft = 400
        self.hop_length = 160

    def _get_wave(self):
        clear_len, inter_len = 0, 0
        while clear_len <= (self.wave_len) or inter_len <= (self.wave_len ):
            id_name = random.sample(self.id_names, 2)
            target_files = random.sample(glob.glob(os.path.join(self.data_path, id_name[0], self.data_format)), 2)
            inter_file = random.choice(glob.glob(os.path.join(self.data_path, id_name[1], self.data_format)))
            refer_wave, _ = librosa.load(target_files[0], sr=self.sampling_rate)
            clear_wave, _ = librosa.load(target_files[1], sr=self.sampling_rate)
            inter_wave, _ = librosa.load(inter_file, sr=self.sampling_rate)
            clear_len = len(clear_wave)
            inter_len = len(inter_wave)
        return refer_wave, clear_wave, inter_wave

    def _stft(self, x):
        spec = librosa.stft(x, n_fft=400, hop_length=160)

    def _dvecwav2mel(self, x):
        spec = librosa.stft(x, n_fft=400, hop_length=160)
        magnitudes = np.abs(spec) ** 2
        mel = np.log10(np.dot(self.mel_basis, magnitudes) + 1e-6)
        norm_mel = self._norm_dvec(mel)
        return norm_mel

    def _mix_wave(self, x1, x2):
        x1_time = random.randrange(len(x1) - self.wave_len)
        x2_time = random.randrange(len(x2) - self.wave_len)
        x1 = x1[x1_time:x1_time+self.wave_len]
        x2 = x1 + x2[x2_time:x2_time+self.wave_len]
        return x1, x2

    def _norm(self, x):
        return np.clip((x / 50.), a_min=-1., a_max=0.) + 1.

    def _denorm(self, x):
        return (x - 1.) * 50.

    def _wav2spec(self, x):
        out = librosa.stft(x, n_fft=self.n_fft)
        out = librosa.amplitude_to_db(abs(out), ref=1., amin=1e-05, top_db=None) - 20.
        out = self._norm(out)
        return out

    def spec2wav(self, x):
        out = self._denorm(x)
        out = librosa.db_to_amplitude((out + 20.), ref=1.)
        out = librosa.griffinlim(out)
        return out

    def train_data(self):
        refer_wave, clear_wave, inter_wave = self._get_wave()
        clear_wave, noicy_wave = self._mix_wave(clear_wave, inter_wave)
        refer_wave = 
        refer_spec = self._dvecwav2mel(refer_wave)
        clear_spec = self._wav2spec(clear_wave)
        noicy_spec = self._wav2spec(noicy_wave)
        return refer_spec, clear_spec, noicy_spec


if __name__ == "__main__":
    get = GetAudio(data_path="/workspace/db/audio/Libri/LibriSpeech/train-clean-360/")
    get.get_wave_dvec()

