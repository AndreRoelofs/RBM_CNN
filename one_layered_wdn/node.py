from torch import nn
from one_layered_wdn.encoder import Encoder
from one_layered_wdn.rv_rbm import RV_RBM


class Node(nn.Module):
    def __init__(self,
                 image_channels,
                 encoder_channels,
                 rbm_visible_units,
                 rbm_hidden_units,
                 rbm_learning_rate,
                 use_relu=False,
                 ):
        super().__init__()
        self.encoder = Encoder(image_channels, encoder_channels, use_relu)
        self.rbm = RV_RBM(
            rbm_visible_units,
            rbm_hidden_units,
            learning_rate=rbm_learning_rate,
            momentum_coefficient=0.0,
            weight_decay=0.00,
            use_cuda=True,
            use_relu=use_relu,
        )
        self.child_networks = []

    def encode(self, x):
        return self.encoder(x)
