from torch import nn
from one_layered_wdn.encoder import Encoder
from one_layered_wdn.rv_rbm import RV_RBM


class Node(nn.Module):
    def __init__(self,
                 image_channels,
                 encoder_channels,
                 encoder_weight_variance,
                 rbm_visible_units,
                 rbm_hidden_units,
                 rbm_learning_rate,
                 use_relu=False,
                 level=0,
                 ):
        super().__init__()
        self.encoder = Encoder(image_channels, encoder_channels, encoder_weight_variance, use_relu=True)
        self.rbm = RV_RBM(
            rbm_visible_units,
            rbm_hidden_units,
            weight_variance=encoder_weight_variance/4,
            use_relu=True,
        )
        self.child_networks = []
        self.level = level
        self.target = None
        self.n_children = 0


    def encode(self, x):
        encoded_data = self.encoder(x)
        return encoded_data
