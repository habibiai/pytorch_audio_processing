from Dataset.MusicDataset import MusicGenreDataset
from Model.cnn import CNNNet
from torch.utils.data.sampler import SubsetRandomSampler


import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np


import warnings
warnings.filterwarnings("ignore")


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
batch_size = 16
validation_split = .2
shuffle_dataset = True
random_seed = 42

# writer = SummaryWriter()
writer = SummaryWriter('runs/genre_classification')


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train(model, train_data, validation_data, loss_fn, optimizer, device, EPOCHS):
    for e in range(EPOCHS):
        print(f"EPOCH {e}/{EPOCHS}")
        train_loss  = 0.0
        val_loss = 0.0
        running_accuracy = 0.0
        total = 0
        for input, target in train_data:

            input, target = input.to(device), target.to(device)

            preds = model(input)
            loss = loss_fn(preds, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        with torch.no_grad():
            model.eval()
            for data in validation_data:
                inputs, outputs = data
                inputs, outputs = inputs.to(device), outputs.to(device)

                predicted_outputs = model(inputs)
                val_loss = loss_fn(predicted_outputs, outputs)

                # The label with the highest value will be our prediction
                _, predicted = torch.max(predicted_outputs, 1)
                val_loss += val_loss.item()
                total += outputs.size(0)
                running_accuracy += (predicted == outputs).sum().item()

                # Calculate validation loss value
        train_loss_value = train_loss / len(validation_data)
        val_loss_value = val_loss / len(validation_data)
        accuracy = (100 * running_accuracy / total)


            # ...log the running loss
        writer.add_scalar('Loss/training loss',
                          train_loss_value,
                          e)
        writer.add_scalar('Loss/validation loss',
                          val_loss_value,
                          e)
        writer.add_scalar('Accuracy/test',
                          accuracy,
                          e)

        print(f"loss : {loss.item()}")
    writer.close()


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    dataset = MusicGenreDataset(ANNOTS_DIR,
                                mel_spectrogram,
                                SAMPLE_RATE,
                                NUM_SAMPLES,
                                device)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)



    cnn = CNNNet(n_classes=N_CLASSES).to(device)
    print(cnn)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    train(cnn, train_loader, validation_loader, loss_fn, optimizer, device, EPOCHS)

    torch.save(cnn.state_dict(), "cnn_net.pth")
    print("cnn netork trained")