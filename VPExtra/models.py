import torch
from torch import nn
from torchaudio import transforms

class FFTClassifier(nn.Module):
    """
    Uses Fast Fourier Transform to transform the input signal
    and then uses fully connected layers for classification

    signal_length:
        length of the input signal
    layer_params:
        list of integers representing the number of neurons in each fully connected layer.
        an additional layer with 1 neuron and sigmoid activation is added at the end
        for binary classification.
    """
    def __init__(self, signal_length, layer_params):
        super().__init__()
        self.fully_connected = nn.Sequential()
        self.layer_params = layer_params
        # rfft output length = (N//2+1) for real input signals
        # after that we concatenate real and imaginary parts
        n0 = 2 * (signal_length // 2 + 1)
        for n in layer_params:
            self.fully_connected.append(nn.Linear(n0, n))
            self.fully_connected.append(nn.ReLU())
            n0 = n
        self.fully_connected.append(nn.Linear(n0, 1))
        self.fully_connected.append(nn.Sigmoid())

    def forward(self, x):
        x = torch.fft.rfft(x) # using rfft for real values input
        x = torch.cat((x.real, x.imag), dim=1)
        x = self.fully_connected(x)
        return x

class MelClassifier(nn.Module):
    """
    Uses Mel Spectrogram to transform the input signal
    and then uses fully connected layers for classification

    signal_length:
        length of the input signal
    """
    def __init__(self, signal_length, layer_params, n_mels = 64, n_hops = 1, sample_rate = 1000):
        super().__init__()
        self.mel_transform = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=signal_length,
            n_mels=n_mels,
            hop_length=signal_length // n_hops
        )
        self.fully_connected = nn.Sequential()
        self.layer_params = layer_params
        n0 = n_mels * (n_hops + 1) # number of mel bins * number of time frames
        for n in layer_params:
            self.fully_connected.append(nn.Linear(n0, n))
            self.fully_connected.append(nn.ReLU())
            n0 = n
        self.fully_connected.append(nn.Linear(n0, 1))
        self.fully_connected.append(nn.Sigmoid())

    def forward(self, x):
        x = self.mel_transform(x)
        x = x.reshape(x.size(0), -1) # flatten the mel spectrogram
        x = self.fully_connected(x)
        return x