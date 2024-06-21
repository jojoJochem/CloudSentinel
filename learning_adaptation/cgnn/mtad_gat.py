import torch
import torch.nn as nn
# from test_tube import HyperOptArgumentParser
from cgnn.modules import (
    Denoising,
    ConvLayer,
    GRULayer,
    Forecasting_Model,
    MHSA,
    AR,
    TemporalcorrelationLayer,
    FeaturecorrelationLayer,

)


class MTAD_GAT(nn.Module):
    """ MTAD-GAT model class.

    :param n_features: Number of input features
    :param window_size: Length of the input sequence
    :param out_dim: Number of features to output
    :param kernel_size: size of kernel to use in the 1-D convolution
    :param feat_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in feat-oriented GAT layer
    :param time_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in time-oriented GAT layer
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param gru_n_layers: number of layers in the GRU layer
    :param gru_hid_dim: hidden dimension in the GRU layer
    :param forecast_n_layers: number of layers in the FC-based Forecasting Model
    :param forecast_hid_dim: hidden dimension in the FC-based Forecasting Model
    :param recon_n_layers: number of layers in the GRU-based Reconstruction Model
    :param recon_hid_dim: hidden dimension in the GRU-based Reconstruction Model
    :param dropout: dropout rate
    :param alpha: negative slope used in the leaky rely activation function
        d_k (int): d_model / n_head
        d_v (int): d_model / n_head
        d_model (int): outputs of dimension
        n_head (int): num of Multi-head
    """

    def __init__(
        self,
        n_features,
        window_size,
        out_dim,
        kernel_size=7,
        feat_gat_embed_dim=None,
        time_gat_embed_dim=None,
        use_gatv2=True,
        gru_n_layers=1,
        gru_hid_dim=150,
        forecast_n_layers=1,
        forecast_hid_dim=150,
        dropout=0.4,
        alpha=0.2,

    ):
        super(MTAD_GAT, self).__init__()

        self.conv = ConvLayer(n_features, kernel_size)

        self.multiheadattention = MHSA(n_features, 3*n_features)
        self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)

        self.ar = AR(window_size)
        self.forecasting_model = Forecasting_Model(gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)

    def forward(self, x):
        gamma = 0.5
        x = Denoising(x)
        h_a = self.ar(x)
        h_a = h_a.view(x.shape[0], -1)

        x = self.conv(x)
        h_feat = FeaturecorrelationLayer(x)
        h_temp = TemporalcorrelationLayer(x)
        h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # (b, n, 3k)   (256,100,38*3)
        h_in = self.multiheadattention(h_cat)
        _, h_end = self.gru(h_in)

        h_end = h_end.view(x.shape[0], -1)   # Hidden state for last timestamp
        predictions = self.forecasting_model(h_end)

        predictions_a = gamma*predictions+(1-gamma)*h_a
        return predictions_a
