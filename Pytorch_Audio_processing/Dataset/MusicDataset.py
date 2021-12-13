from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import os
import torch




class MusicGenreDataset(Dataset):

    def __init__(self,annot_dir,transformation,target_sample_rate, num_samples, device):

        self.annotations = pd.read_csv(annot_dir)
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.device = device




    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal,sr = torchaudio.load(audio_sample_path)
        # signal -> (num_ch, samples) -> (2,16000) -> (1,16000)
        signal = self._resample_if_necessary(signal,sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)

        # print("path", audio_sample_path)
        # print("label", label)

        return signal,label


    def _cut_if_necessary(self,signal):
        # signal -> Tensor -> (1, num_samples)
        if signal.shape[1]> self.num_samples:
            signal = signal[:,:self.num_samples]
        return signal


    def _right_pad_if_necessary(self,signal):
         #[1 2 3 2 2] -> [1 2 3 2 2 0 0 0 ]
        signal_length = signal.shape[1]
        if signal_length  < self.num_samples:
            num_missing_samples = self.num_samples - signal_length
            last_dim_padding  = (0,num_missing_samples)
            signal = torch.nn.functional.pad(signal,last_dim_padding)
        return signal


    def _resample_if_necessary(self,signal, sr):
        resampler = torchaudio.transforms.Resample(sr,self.target_sample_rate)
        signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self,signal):
        if signal.shape[0]>1:
            signal = torch.mean(signal,dim = 0,keepdim=True)
        return signal


    def _get_audio_sample_path(self,index):
        return self.annotations.iloc[index,1]

    def _get_audio_sample_label(self,index):
        return self.annotations.iloc[index,2]



if __name__ == '__main__':


    # print("v",str(torchaudio.get_audio_backend()))

    annot_dir = "genre_data.csv"
    sample_rate = 22050
    n_samples = 22050
    n_fft = 1024
    hop_length = 512
    n_mels = 64

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f" device is{device}")


    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = sample_rate,
        n_fft= n_fft,
        hop_length= hop_length,
        n_mels=n_mels
    )

    usd = MusicGenreDataset(
        annot_dir,
        transformation=mel_spectrogram,
        target_sample_rate = sample_rate,
        num_samples=n_samples,
        device=device)




    print(f"There are {len(usd)} samples in the dataset")
    signal,label = usd[0]
    print("rnd")