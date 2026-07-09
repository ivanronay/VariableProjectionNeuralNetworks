import torch
from torch import nn

class FTTClassifier(nn.Module):
    """
    Uses Fast Fourier Transform to transform the input signal
    and then uses fully connected layers for classification

    signal_length:
        length of the input signal
    layer_params:
        list of integers representing the number of neurons in each fully connected layer.
        an additional layer with 1 neuron and sigmoid activation is added at the end for binary classification.
    """
    def __init__(self, signal_length, layer_params):
        super().__init__()
        self.fully_connected = nn.Sequential()
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
    
    # train method calls them but they are not needed for this model
    def train_mode(self):
        pass
    
    def test_mode(self):
        pass