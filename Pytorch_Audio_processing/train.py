
from Dataset.MusicDataset import MusicGenreDataset
from Model.cnn import CNNNet

import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter





BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001

ANNOTS_DIR = 'Dataset/genre_data.csv'
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64
N_CLASSES = 10

writer = SummaryWriter()
writer = SummaryWriter('runs/fashion_mnist_experiment_1')




def create_data_loader(train_data,batch_size):

    train_dataloader = DataLoader(train_data,batch_size=batch_size)
    return train_dataloader

def train(model, data_loader,loss_fn,optimizer, device, EPOCHS):

    for e in range(EPOCHS):
        print(f"EPOCH {e}/{EPOCHS}")
        for input,target in data_loader:
            input, target = input.to(device),target.to(device)

            preds = model(input)
            loss = loss_fn(preds,target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"loss : {loss.item()}")

if __name__ =='__main__':
    if torch.cuda.is_available():
        device='cuda'
    else:
        device = 'cpu'
    print(f"Using {device}")



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

    dataloader = create_data_loader(dataset,BATCH_SIZE)

    train_size = int(0.8 * len(dataloader))
    test_size = len(dataloader) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataloader, [train_size, test_size])

    cnn = CNNNet(n_classes=N_CLASSES).to(device)
    print(cnn)


    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(),
                                 lr = LEARNING_RATE)


    train(cnn,train_dataset,loss_fn,optimizer,device,EPOCHS)

    torch.save(cnn.state_dict(),"cnn_net.pth")
    print("cnn netork trained")