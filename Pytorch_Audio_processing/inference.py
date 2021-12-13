
from Dataset.MusicDataset import MusicGenreDataset
from Model.cnn import CNNNet

import torch
import torchaudio
from torch.utils.data import DataLoader



class_mapping = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",

]

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

ANNOTS_DIR = 'Dataset/genre_data.csv'
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64
N_CLASSES = 10
model_path = "cnn_net.pth"


def predict(model, input, target, class_mapping):
    model.eval()

    with torch.no_grad():
        preds = model(input)  # Tensor(1, 10) -> [[0.1 , 0.01, ... , 0.6]]
        predicted_index = preds[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":

    if torch.cuda.is_available():
        device='cuda'
    else:
        device = 'cpu'
    print(f"Using {device}")

    cnn = CNNNet(N_CLASSES)
    state_dict = torch.load(model_path)
    cnn.load_state_dict(state_dict)


    # Load Music genre dataset

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = N_FFT,
        hop_length = HOP_LENGTH,
        n_mels = N_MELS
    )

    dataset = MusicGenreDataset(ANNOTS_DIR,
                                mel_spectrogram,
                                SAMPLE_RATE,
                                NUM_SAMPLES,
                                device)


    input, target = dataset[0][0], dataset[0][1] # [batch_size, num_ch, frames, time]
    input.unsqueeze_(0)

    predicted, expected = predict(cnn, input, target, class_mapping)

    print(f"Predicted : {predicted}, Expected : {expected}")







