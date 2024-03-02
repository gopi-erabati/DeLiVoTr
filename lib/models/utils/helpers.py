# Adapted from delight_modules in Fairseq

import time
import torch
from torch import nn
import torch.nn.functional as F


class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(BatchNorm, self).__init__()
        self.layer = nn.BatchNorm1d(num_features=num_features, eps=eps,
                                    affine=affine)

    def forward(self, x):
        if x.dim() == 3:
            bsz, seq_len, feature_size = x.size()
            out = self.layer(x.view(-1, feature_size))
            return out.contiguous().view(bsz, seq_len, -1)
        else:
            return self.layer(x)


norm_layer_list = [
    'gn', 'bn', 'ln'
]


def get_norm_layer(name, out_features, num_groups=1, eps=1e-5, affine=True):
    if name == 'gn' and num_groups == 1:
        name = 'bn'

    if name == 'bn':
        return BatchNorm(num_features=out_features, eps=eps, affine=affine)
    elif name == 'ln':
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(out_features, eps, affine)
        except:
            return nn.LayerNorm(out_features, eps=eps,
                                elementwise_affine=affine)
    elif name == 'gn':
        return nn.GroupNorm(num_groups=num_groups, num_channels=out_features,
                            eps=eps, affine=affine)
    else:
        print_error_message(
            'Supported normalization functions: {}'.format(norm_layer_list))
        return None


class GELU(torch.nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return torch.nn.functional.gelu(x)


class Swish(torch.nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


activation_list = [
    'relu', 'leaky', 'selu', 'elu', 'celu', 'prelu', 'sigmoid', 'tanh', 'gelu',
    'swish'
]


def get_activation_layer(name):
    if name == 'relu':
        return nn.ReLU(inplace=False)
    elif name == 'leaky':
        return nn.LeakyReLU(negative_slope=0.1, inplace=False)
    elif name == 'selu':
        return nn.SELU(inplace=True)
    elif name == 'elu':
        return nn.ELU(inplace=True)
    elif name == 'celu':
        return nn.CELU(inplace=True)
    elif name == 'prelu':
        return nn.PReLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'gelu':
        return GELU()
    elif name == 'swish':
        return Swish()
    else:
        print_error_message(
            'Supported activation functions: {}'.format(activation_list))
        return None


def softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)

text_colors = {
    'logs': '\033[34m',  # 033 is the escape code and 34 is the color code
    'info': '\033[32m',
    'warning': '\033[33m',
    'error': '\033[31m',
    'bold': '\033[1m',
    'end_color': '\033[0m'
}


def get_curr_time_stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def print_error_message(message):
    time_stamp = get_curr_time_stamp()
    error_str = text_colors['error'] + text_colors['bold'] + 'ERROR  ' + \
                text_colors['end_color']
    print('{} - {} - {}'.format(time_stamp, error_str, message))
    print('{} - {} - {}'.format(time_stamp, error_str, 'Exiting!!!'))
    exit(-1)


def print_log_message(message):
    time_stamp = get_curr_time_stamp()
    log_str = text_colors['logs'] + text_colors['bold'] + 'LOGS   ' + \
              text_colors['end_color']
    print('{} - {} - {}'.format(time_stamp, log_str, message))


def print_warning_message(message):
    time_stamp = get_curr_time_stamp()
    warn_str = text_colors['warning'] + text_colors['bold'] + 'WARNING' + \
               text_colors['end_color']
    print('{} - {} - {}'.format(time_stamp, warn_str, message))


def print_info_message(message):
    time_stamp = get_curr_time_stamp()
    info_str = text_colors['info'] + text_colors['bold'] + 'INFO   ' + \
               text_colors['end_color']
    print('{} - {} - {}'.format(time_stamp, info_str, message))


def print_dash_line():
    print(text_colors['error'] + text_colors['bold'] + '=' * 100 + text_colors[
        'end_color'])


def print_header(header):
    print_dash_line()
    print(text_colors['info'] + text_colors['bold'] + '=' * 50 + str(header) +
          text_colors['end_color'])
    print_dash_line()


def print_header_minor(header):
    print(
        text_colors['warning'] + text_colors['bold'] + '=' * 25 + str(header) +
        text_colors['end_color'])
