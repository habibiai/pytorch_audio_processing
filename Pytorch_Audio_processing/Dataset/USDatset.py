from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import os




class UrbanSoundDataset(Dataset):

    def __init__(self,annot_dir,audio_dir):

        self.annotations = pd.read_csv(annot_dir)
        self.audio_dir = audio_dir


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal,sr = torchaudio.load(audio_sample_path)
        return signal,label



    def _get_audio_sample_path(self,index):
        fold = f'fold{self.annotations.iloc[index,5]}'
        path = os.path.join(self.audio_dir,fold,self.annotations.iloc[index,0])
        return path
    def _get_audio_sample_label(self,index):
        return self.annotations.iloc[index,6]



if __name__ == '__main__':
    annot_dir = ""
    audio_dir = ""
    usd = UrbanSoundDataset(annot_dir,audio_dir)

    print(f"There are {len(usd)} samples in the dataset")
    signal,label = usd[0]