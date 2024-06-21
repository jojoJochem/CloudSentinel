import torch
import torch.nn as nn
import numpy as np


class ConvLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation
    """

    def __init__(self, n_features, kernel_size=7):
        super(ConvLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)  # Permute back


def TemporalcorrelationLayer(x):
    use_cuda = True  #
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    matrix_all = []
    y = x.data.cpu().numpy()

    for k in range(y.shape[0]):
        data = y[k]
        matrix = np.zeros((data.shape[0], data.shape[0]))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                matrix[i][j] = np.correlate(data[i, :], data[j, :])

        matrix = matrix / data.shape[0]
        matrix_all.append(matrix)
    attention = torch.from_numpy(np.array(matrix_all))
    attention = attention.to(dtype=torch.float32)

    attention = attention.to(device)
    h = torch.sigmoid(torch.matmul(attention, x))  # (b, n, k)

    return h


def FeaturecorrelationLayer(x):
    # print(f'x={x.shape}')
    use_cuda = True  #
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    matrix_all = []
    y = x.data.cpu().numpy()

    for k in range(y.shape[0]):
        data = y[k]
        matrix = np.zeros((data.shape[1], data.shape[1]))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if (i <= j):
                    matrix[i][j] = np.inner(data[:, i], data[:, j])
                else:
                    break
        matrix = matrix / data.shape[0]
        matrix_all.append(matrix)
    attention = torch.from_numpy(np.array(matrix_all))
    attention = attention.to(dtype=torch.float32)
    attention = attention.to(device)
    # print(attention.shape)
    h = torch.sigmoid(torch.matmul(attention, x.permute(0, 2, 1)))
    # print(f'h={h.shape}')
    return h.permute(0, 2, 1)


class GRULayer(nn.Module):
    """Gated Recurrent Unit (GRU) Layer
    :param in_dim: number of input features
    :param hid_dim: hidden size of the GRU
    :param n_layers: number of layers in GRU
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(GRULayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        out, h = self.gru(x)
        out, h = out[-1, :, :], h[-1, :, :]  # Extracting from last layer
        return out, h


class Forecasting_Model(nn.Module):
    """Forecasting model (fully-connected network)
    :param in_dim: number of input features
    :param hid_dim: hidden size of the FC network
    :param out_dim: number of output features
    :param n_layers: number of FC layers
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(Forecasting_Model, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)

        return self.layers[-1](x)


def Denoising(train):
    use_cuda = True
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    io_all = []
    for i in range(train.shape[0]):
        data = train[i]
        data = data.data.cpu().numpy()
        io_time = []
        for j in range(data.shape[1]):
            x = data[:, j]
            # x = x.data.cpu().numpy()
            f = np.fft.rfft(x)
            yf_abs = np.abs(f)
            indices = yf_abs > yf_abs.mean()  # filter out those value under 300
            yf_clean = indices * f
            new_f_clean = np.fft.irfft(yf_clean)
            io_time.append(new_f_clean)
        io_time = np.array(io_time)
        io_all.append(io_time)
    io_all = np.array(io_all)
    io_all = torch.from_numpy(np.array(io_all))
    io_all = io_all.to(dtype=torch.float32)
    io_all = io_all.permute(0, 2, 1)
    io_all = io_all.to(device)
    return io_all


class AR(nn.Module):

    def __init__(self, window):
        super(AR, self).__init__()
        self.linear = nn.Linear(window, 1)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)

        return x


class MHSA(nn.Module):
    def __init__(self, num_heads, dim):
        super().__init__()

        # Q, K, V 转换矩阵，这里假设输入和输出的特征维度相同
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.num_heads = num_heads

    def forward(self, x):
        # print(x.shape)
        B, N, C = x.shape
        # 生成转换矩阵并分多头
        q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        # 点积得到attention score
        attn = q @ k.transpose(2, 3) * (x.shape[-1] ** -0.5)
        attn = attn.softmax(dim=-1)
        # 乘上attention score并输出
        v = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        # print(v.shape)
        return v
