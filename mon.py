import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MONSingleFc(nn.Module):
    """ Simple MON linear class, just a single full multiply. """

    def __init__(self, in_dim, out_dim, m=1.0):
        super().__init__()
        self.U = nn.Linear(in_dim, out_dim)
        self.A = nn.Linear(out_dim, out_dim, bias=False)
        self.B = nn.Linear(out_dim, out_dim, bias=False)
        self.m = m

    def x_shape(self, n_batch):
        return (n_batch, self.U.in_features)

    def z_shape(self, n_batch):
        return ((n_batch, self.A.in_features),)

    def forward(self, x, *z):
        return (self.U(x) + self.multiply(*z)[0],)

    def bias(self, x):
        return (self.U(x),)

    def multiply(self, *z):
        ATAz = self.A(z[0]) @ self.A.weight
        z_out = (1 - self.m) * z[0] - ATAz + self.B(z[0]) - z[0] @ self.B.weight
        return (z_out,)

    def multiply_transpose(self, *g):
        ATAg = self.A(g[0]) @ self.A.weight
        g_out = (1 - self.m) * g[0] - ATAg - self.B(g[0]) + g[0] @ self.B.weight
        return (g_out,)

    def init_inverse(self, alpha, beta):
        I = torch.eye(self.A.weight.shape[0], dtype=self.A.weight.dtype,
                      device=self.A.weight.device)
        W = (1 - self.m) * I - self.A.weight.T @ self.A.weight + self.B.weight - self.B.weight.T
        self.Winv = torch.inverse(alpha * I + beta * W)

    def inverse(self, *z):
        return (z[0] @ self.Winv.transpose(0, 1),)

    def inverse_transpose(self, *g):
        return (g[0] @ self.Winv,)


class MONReLU(nn.Module):
    def forward(self, *z):
        return tuple(F.relu(z_) for z_ in z)

    def derivative(self, *z):
        return tuple((z_ > 0).type_as(z[0]) for z_ in z)




# Convolutional layers w/ FFT-based inverses

def fft_to_complex_matrix(x):
    """ Create matrix with [a -b; b a] entries for complex numbers. """
    x_stacked = torch.stack((x, torch.flip(x, (4,))), dim=5).permute(2, 3, 0, 4, 1, 5)
    x_stacked[:, :, :, 0, :, 1] *= -1
    return x_stacked.reshape(-1, 2 * x.shape[0], 2 * x.shape[1])


def fft_to_complex_vector(x):
    """ Create stacked vector with [a;b] entries for complex numbers"""
    return x.permute(2, 3, 0, 1, 4).reshape(-1, x.shape[0], x.shape[1] * 2)


def init_fft_conv(weight, hw):
    """ Initialize fft-based convolution.

    Args:
        weight: Pytorch kernel
        hw: (height, width) tuple
    """
    px, py = (weight.shape[2] - 1) // 2, (weight.shape[3] - 1) // 2
    kernel = torch.flip(weight, (2, 3))
    kernel = F.pad(F.pad(kernel, (0, hw[0] - weight.shape[2], 0, hw[1] - weight.shape[3])),
                   (0, py, 0, px), mode="circular")[:, :, py:, px:]
    return fft_to_complex_matrix(torch.rfft(kernel, 2, onesided=False))


def fft_conv(x, w_fft, transpose=False):
    """ Perhaps FFT-based circular convolution.

    Args:
        x: (B, C, H, W) tensor
        w_fft: conv kernel processed by init_fft_conv
        transpose: flag of whether to transpose convolution
    """
    x_fft = fft_to_complex_vector(torch.rfft(x, 2, onesided=False))
    wx_fft = x_fft.bmm(w_fft.transpose(1, 2)) if not transpose else x_fft.bmm(w_fft)
    wx_fft = wx_fft.view(x.shape[2], x.shape[3], wx_fft.shape[1], -1, 2).permute(2, 3, 0, 1, 4)
    return torch.irfft(wx_fft, 2, onesided=False)


class MONSingleConv(nn.Module):
    """ MON class with a single 3x3 (circular) convolution """

    def __init__(self, in_channels, out_channels, shp, kernel_size=3, m=1.0):
        super().__init__()
        self.U = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.A = nn.Conv2d(out_channels, out_channels, kernel_size, bias=False)
        self.g = nn.Parameter(torch.tensor(1.))
        self.h = nn.Parameter(torch.tensor(1.))
        self.B = nn.Conv2d(out_channels, out_channels, kernel_size, bias=False)
        self.pad = 4 * ((kernel_size - 1) // 2,)
        self.shp = shp
        self.m = m

    def cpad(self, x):
        return F.pad(x, self.pad, mode="circular")

    def uncpad(self, x):
        return x[:, :, 2 * self.pad[0]:-2 * self.pad[1], 2 * self.pad[2]:-2 * self.pad[3]]

    def x_shape(self, n_batch):
        return (n_batch, self.U.in_channels, self.shp[0], self.shp[1])

    def z_shape(self, n_batch):
        return ((n_batch, self.A.in_channels, self.shp[0], self.shp[1]),)

    def forward(self, x, *z):
        # circular padding is broken in PyTorch
        return (F.conv2d(self.cpad(x), self.U.weight, self.U.bias) + self.multiply(*z)[0],)

    def bias(self, x):
        return (F.conv2d(self.cpad(x), self.U.weight, self.U.bias),)

    def multiply(self, *z):
        A = self.A.weight / self.A.weight.view(-1).norm()
        B = self.h * self.B.weight / self.B.weight.view(-1).norm()
        Az = F.conv2d(self.cpad(z[0]), A)
        ATAz = self.uncpad(F.conv_transpose2d(self.cpad(Az), A))
        Bz = F.conv2d(self.cpad(z[0]), B)
        BTz = self.uncpad(F.conv_transpose2d(self.cpad(z[0]), B))
        z_out = (1 - self.m) * z[0] - self.g * ATAz + Bz - BTz
        return (z_out,)

    def multiply_transpose(self, *g):
        A = self.A.weight / self.A.weight.view(-1).norm()
        B = self.h * self.B.weight / self.B.weight.view(-1).norm()
        Ag = F.conv2d(self.cpad(g[0]), A)
        ATAg = self.uncpad(F.conv_transpose2d(self.cpad(Ag), A))
        Bg = F.conv2d(self.cpad(g[0]), B)
        BTg = self.uncpad(F.conv_transpose2d(self.cpad(g[0]), B))
        g_out = (1 - self.m) * g[0] - self.g * ATAg - Bg + BTg
        return (g_out,)

    def init_inverse(self, alpha, beta):
        A = self.A.weight / self.A.weight.view(-1).norm()
        B = self.h * self.B.weight / self.B.weight.view(-1).norm()
        Afft = init_fft_conv(A, self.shp)
        Bfft = init_fft_conv(B, self.shp)
        I = torch.eye(Afft.shape[1], dtype=Afft.dtype,
                      device=Afft.device)[None, :, :]
        self.Wfft = (1 - self.m) * I - self.g * Afft.transpose(1, 2) @ Afft + Bfft - Bfft.transpose(1, 2)
        self.Winv = torch.inverse(alpha * I + beta * self.Wfft)

    def inverse(self, *z):
        return (fft_conv(z[0], self.Winv),)

    def inverse_transpose(self, *g):
        return (fft_conv(g[0], self.Winv, transpose=True),)


class MONBorderReLU(nn.Module):
    def __init__(self, border=1):
        super().__init__()
        self.border = border

    def forward(self, *z):
        zn = tuple(F.relu(z_) for z_ in z)
        for i in range(len(zn)):
            zn[i][:, :, :self.border, :] = 0
            zn[i][:, :, -self.border:, :] = 0
            zn[i][:, :, :, :self.border] = 0
            zn[i][:, :, :, -self.border:] = 0
        return zn

    def derivative(self, *z):
        return tuple((z_ > 0).type_as(z[0]) for z_ in z)


class MONMultiConv(nn.Module):
    def __init__(self, in_channels, conv_channels, image_size, kernel_size=3, m=1.0):
        super().__init__()
        self.pad = 4 * ((kernel_size - 1) // 2,)
        self.conv_shp = tuple((image_size - 2 * self.pad[0]) // 2 ** i + 2 * self.pad[0]
                              for i in range(len(conv_channels)))
        self.m = m

        # create convolutional layers
        self.U = nn.Conv2d(in_channels, conv_channels[0], kernel_size)
        self.A0 = nn.ModuleList([nn.Conv2d(c, c, kernel_size, bias=False) for c in conv_channels])
        self.B0 = nn.ModuleList([nn.Conv2d(c, c, kernel_size, bias=False) for c in conv_channels])
        self.A_n0 = nn.ModuleList([nn.Conv2d(c1, c2, kernel_size, bias=False, stride=2)
                                   for c1, c2 in zip(conv_channels[:-1], conv_channels[1:])])

        self.g = nn.ParameterList([nn.Parameter(torch.tensor(1.)) for _ in range(len(conv_channels))])
        self.gn = nn.ParameterList([nn.Parameter(torch.tensor(1.)) for _ in range(len(conv_channels) - 1)])
        self.h = nn.ParameterList([nn.Parameter(torch.tensor(1.)) for _ in range(len(conv_channels))])



        self.S_idx = list()
        self.S_idxT = list()
        for n in self.conv_shp:
            p = n // 2
            q = n
            idxT = list()
            _idx = [[j + (i - 1) * p for i in range(1, q + 1)] for j in range(1, p + 1)]
            for i in _idx:
                for j in i:
                    idxT.append(j - 1)
            _idx = [[j + (i - 1) * p + p * q for i in range(1, q + 1)] for j in range(1, p + 1)]
            for i in _idx:
                for j in i:
                    idxT.append(j - 1)
            idx = list()
            _idx = [[j + (i - 1) * q for i in range(1, p + 1)] for j in range(1, q + 1)]
            for i in _idx:
                for j in i:
                    idx.append(j - 1)
            _idx = [[j + (i - 1) * q + p * q for i in range(1, p + 1)] for j in range(1, q + 1)]
            for i in _idx:
                for j in i:
                    idx.append(j - 1)
            self.S_idx.append(idx)
            self.S_idxT.append(idxT)

    def A(self, i):
        return torch.sqrt(self.g[i]) * self.A0[i].weight / self.A0[i].weight.view(-1).norm()

    def A_n(self, i):
        return torch.sqrt(self.gn[i]) * self.A_n0[i].weight / self.A_n0[i].weight.view(-1).norm()

    def B(self, i):
        return self.h[i] * self.B0[i].weight / self.B0[i].weight.view(-1).norm()

    def cpad(self, x):
        return F.pad(x, self.pad, mode="circular")

    def uncpad(self, x):
        return x[:, :, 2 * self.pad[0]:-2 * self.pad[1], 2 * self.pad[2]:-2 * self.pad[3]]

    def zpad(self, x):
        return F.pad(x, (0, 1, 0, 1))

    def unzpad(self, x):
        return x[:, :, :-1, :-1]

    def unstride(self, x):
        x[:, :, :, -1] += x[:, :, :, 0]
        x[:, :, -1, :] += x[:, :, 0, :]
        return x[:, :, 1:, 1:]

    def x_shape(self, n_batch):
        return (n_batch, self.U.in_channels, self.conv_shp[0], self.conv_shp[0])

    def z_shape(self, n_batch):
        return tuple((n_batch, self.A0[i].in_channels, self.conv_shp[i], self.conv_shp[i])
                     for i in range(len(self.A0)))

    def forward(self, x, *z):
        z_out = self.multiply(*z)
        bias = self.bias(x)
        return tuple([z_out[i] + bias[i] for i in range(len(self.A0))])

    def bias(self, x):
        z_shape = self.z_shape(x.shape[0])
        n = len(self.A0)

        b_out = [self.U(self.cpad(x))]
        for i in range(n - 1):
            b_out.append(torch.zeros(z_shape[i + 1], dtype=self.A0[0].weight.dtype,
                   device=self.A0[0].weight.device))
        return tuple(b_out)

    def multiply(self, *z):

        def multiply_zi(z1, A1, B1, A1_n=None, z0=None, A2_n=None):
            Az1 = F.conv2d(self.cpad(z1), A1)
            A1TA1z1 = self.uncpad(F.conv_transpose2d(self.cpad(Az1), A1))
            B1z1 = F.conv2d(self.cpad(z1), B1)
            B1Tz1 = self.uncpad(F.conv_transpose2d(self.cpad(z1), B1))
            out = (1 - self.m) * z1 - A1TA1z1 + B1z1 - B1Tz1
            if A2_n is not None:
                A2_nz1 = F.conv2d(self.cpad(z1), A2_n, stride=2)
                A2_nTA2_nz1 = self.unstride(F.conv_transpose2d(A2_nz1,
                                                               A2_n, stride=2))
                out -= A2_nTA2_nz1
            if A1_n is not None:
                A1_nz0 = self.zpad(F.conv2d(self.cpad(z0), A1_n, stride=2))
                A1TA1_nz0 = self.uncpad(F.conv_transpose2d(self.cpad(A1_nz0), A1))
                out -= 2 * A1TA1_nz0
            return out

        n = len(self.A0)
        z_out = [multiply_zi(z[0], self.A(0), self.B(0), A2_n=self.A_n(0))]
        for i in range(1, n - 1):
            z_out.append(multiply_zi(z[i], self.A(i), self.B(i),
                                     A1_n=self.A_n(i - 1), z0=z[i - 1], A2_n=self.A_n(i)))
        z_out.append(multiply_zi(z[n - 1], self.A(n - 1), self.B(n - 1),
                                 A1_n=self.A_n(n - 2), z0=z[n - 2]))

        return tuple(z_out)

    def multiply_transpose(self, *g):

        def multiply_zi(z1, A1, B1, z2=None, A2_n=None, A2=None):
            Az1 = F.conv2d(self.cpad(z1), A1)
            A1TA1z1 = self.uncpad(F.conv_transpose2d(self.cpad(Az1), A1))
            B1z1 = F.conv2d(self.cpad(z1), B1)
            B1Tz1 = self.uncpad(F.conv_transpose2d(self.cpad(z1), B1))
            out = (1 - self.m) * z1 - A1TA1z1 - B1z1 + B1Tz1
            if A2_n is not None:
                A2z2 = F.conv2d(self.cpad(z2), A2)
                A2_nTA2z2 = self.unstride(F.conv_transpose2d(self.unzpad(A2z2),
                                                             A2_n, stride=2))

                out -= 2 * A2_nTA2z2

                A2_nz1 = F.conv2d(self.cpad(z1), A2_n, stride=2)
                A2_nTA2_nz1 = self.unstride(F.conv_transpose2d(A2_nz1,
                                                               A2_n, stride=2))

                out -= A2_nTA2_nz1

            return out

        n = len(self.A0)
        g_out = []
        for i in range(n - 1):
            g_out.append(multiply_zi(g[i], self.A(i), self.B(i), z2=g[i + 1], A2_n=self.A_n(i), A2=self.A(i + 1)))
        g_out.append(multiply_zi(g[n - 1], self.A(n - 1), self.B(n - 1)))

        return g_out

    def init_inverse(self, alpha, beta):
        n = len(self.A0)
        conv_fft_A = [init_fft_conv(self.A(i), (self.conv_shp[i], self.conv_shp[i]))
                      for i in range(n)]
        conv_fft_B = [init_fft_conv(self.B(i), (self.conv_shp[i], self.conv_shp[i]))
                      for i in range(n)]

        conv_fft_A_n = [init_fft_conv(self.A_n(i - 1), (self.conv_shp[i - 1], self.conv_shp[i - 1]))
                        for i in range(1, n)]

        I = [torch.eye(2 * self.A0[i].weight.shape[1], dtype=self.A0[i].weight.dtype,
                       device=self.A0[i].weight.device)[None, :, :] for i in range(n)]

        D1 = [(alpha + beta - beta * self.m) * I[i] \
              - beta * conv_fft_A[i].transpose(1, 2) @ conv_fft_A[i] \
              + beta * conv_fft_B[i] - beta * conv_fft_B[i].transpose(1, 2)
              for i in range(n - 1)]

        self.D1inv = [torch.inverse(D) for D in D1]

        self.D2 = [np.sqrt(-beta) * conv_fft_A_n[i] for i in range(n - 1)]

        G = [(self.D2[i] @ self.D1inv[i] @ self.D2[i].transpose(1, 2))[self.S_idx[i]] for i in range(n - 1)]
        S = [G[i][:self.conv_shp[i] ** 2 // 4]
             + G[i][self.conv_shp[i] ** 2 // 4:self.conv_shp[i] ** 2 // 2]
             + G[i][self.conv_shp[i] ** 2 // 2:3 * self.conv_shp[i] ** 2 // 4]
             + G[i][3 * self.conv_shp[i] ** 2 // 4:]
             for i in range(n - 1)]
        Hinv = [torch.eye(s.shape[1], device=s.device) + 0.25 * s for s in S]
        self.H = [torch.inverse(hinv).float() for hinv in Hinv]

        Wn = (1 - self.m) * I[n - 1] \
             - conv_fft_A[n - 1].transpose(1, 2) @ conv_fft_A[n - 1] \
             + conv_fft_B[n - 1] - conv_fft_B[n - 1].transpose(1, 2)

        self.Wn_inv = torch.inverse(alpha * I[n - 1] + beta * Wn)

        self.beta = beta

    def apply_inverse_conv(self, z, i):
        z0_fft = fft_to_complex_vector(torch.rfft(z, 2, onesided=False))
        y0 = 0.5 * z0_fft.bmm((self.D2[i] @ self.D1inv[i]).transpose(1, 2))[self.S_idx[i]]
        n = self.conv_shp[i]
        y1 = y0[:n ** 2 // 4] + y0[n ** 2 // 4:n ** 2 // 2] + y0[n ** 2 // 2:3 * n ** 2 // 4] + y0[3 * n ** 2 // 4:]
        y2 = y1.bmm(self.H[i].transpose(1, 2))
        y3 = y2.repeat(4, 1, 1)
        y4 = y3[self.S_idxT[i]]
        y5 = 0.5 * y4.bmm(self.D2[i] @ self.D1inv[i].transpose(1, 2))
        x0 = z0_fft.bmm(self.D1inv[i].transpose(1, 2)) - y5
        x0 = x0.view(n, n, x0.shape[1], -1, 2).permute(2, 3, 0, 1, 4)
        x0 = torch.irfft(x0, 2, onesided=False)
        return x0

    def apply_inverse_conv_transpose(self, g, i):
        g0_fft = fft_to_complex_vector(torch.rfft(g, 2, onesided=False))
        y0 = 0.5 * g0_fft.bmm(self.D1inv[i] @ self.D2[i].transpose(1, 2))[self.S_idx[i]]
        n = self.conv_shp[i]
        y1 = y0[:n ** 2 // 4] + y0[n ** 2 // 4:n ** 2 // 2] + y0[n ** 2 // 2:3 * n ** 2 // 4] + y0[3 * n ** 2 // 4:]
        y2 = y1.bmm(self.H[i])
        y3 = y2.repeat(4, 1, 1)
        y4 = y3[self.S_idxT[i]]
        y5 = 0.5 * y4.bmm(self.D2[i] @ self.D1inv[i])
        x0 = g0_fft.bmm(self.D1inv[i]) - y5
        x0 = x0.view(n, n, x0.shape[1], -1, 2).permute(2, 3, 0, 1, 4)
        x0 = torch.irfft(x0, 2, onesided=False)
        return x0

    def inverse(self, *z):
        n = len(self.A0)
        x = [self.apply_inverse_conv(z[0], 0)]
        for i in range(n - 1):
            A_nx0 = self.zpad(F.conv2d(self.cpad(x[-1]), self.A_n(i), stride=2))
            ATA_nx0 = self.uncpad(F.conv_transpose2d(self.cpad(A_nx0), self.A(i + 1)))
            xn = -self.beta * 2 * ATA_nx0
            if i < n - 2:
                x.append(self.apply_inverse_conv(z[i + 1] - xn, i + 1))
            else:
                x.append(fft_conv(z[i + 1] - xn, self.Wn_inv))

        return tuple(x)

    def inverse_transpose(self, *g):
        n = len(self.A0)

        x = [fft_conv(g[-1], self.Wn_inv, transpose=True)]
        for i in range(n - 2, -1, -1):
            A2x2 = F.conv2d(self.cpad(x[-1]), self.A(i + 1))
            A2_NTA2x2 = self.unstride(F.conv_transpose2d(self.unzpad(A2x2),
                                                         self.A_n(i), stride=2))
            xp = -self.beta * 2 * A2_NTA2x2
            x.append(self.apply_inverse_conv_transpose(g[i] - xp, i))
        x.reverse()
        return tuple(x)