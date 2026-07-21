import torch
from torch import nn

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
    layer_params:
        list of integers representing the number of neurons in each fully connected layer.
        an additional layer with 1 neuron and sigmoid activation is added at the end
        for binary classification.
    n_mels:
        number of mel bins to use for the mel spectrogram
    n_hops:
        number of time frames to use for the mel spectrogram
    sample_rate:
        sample rate of the input signal
    """
    def __init__(self, signal_length, layer_params, n_mels = 64, n_hops = 1, sample_rate = 1000):
        super().__init__()
        from torchaudio import transforms
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

class CWTClassifierSNN(nn.Module):
    """
    Uses Spiking Neural Network for classification
    """
    def __init__(self, layer_params, vp_params, beta=0.9):
        super().__init__()
        import snntorch as snn
        from VPBase import CWTLayer as cwt
        self.vp_layer = cwt.vp_layer(vp_params)
        self.layers = nn.Sequential()
        self.layer_params = layer_params
        
        if vp_params['vp_target'] == 0:
            n0 = vp_params['vp_latent_dim']
        else: 
            n0 = vp_params['input_length']

        for n in layer_params:
            self.layers.append(nn.Linear(n0, n))
            self.layers.append(snn.Leaky(beta=beta))
            n0 = n

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.vp_layer.Phi = None
            self.vp_layer.Phip = None
        else:
            psi, _, _ = self.vp_layer.ada(self.vp_layer.weight)
            self.vp_layer.Phi = psi
            self.vp_layer.Phip = torch.linalg.pinv(psi)

    def forward(self, x):
        x = self.vp_layer(x)
        torch.squeeze(x)
        if 1 == len(x.shape): x = x.unsqueeze(0) # BS = 1
        x = self.layers(x)
        # TODO: implement SNN specific behaviour
        return x