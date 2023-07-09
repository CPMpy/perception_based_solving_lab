import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import cpmpy as cp

def make_fc_layers(cfg, in_dim=512, activ=nn.ReLU, p=0.0, bn=False, out=nn.Softmax):
    """Helper function to make layers of a Fully-Connected Neural Network. 

    Args:
        cfg (List[Tuple[Int,Int]] or List[Int]): Each tuple defines a (dimension_input, dimension_output) pair of the layer. In case of a single Int, dimension_output of the last layer is treated as dimension_input.
        in_dim (int, optional): dimension of the input layer of the neural network. Defaults to 512.
        activ (Type[nn.Module], optional): activation function constructor. Defaults to nn.ReLU.
        p (float, optional): Dropout parameter. Defaults to 0.0 (disables dropout).
        bn (bool, optional): toggle Batch Normalization. Defaults to False.
        out (Type[nn.Module], optional): output layer function constructor. Defaults to nn.LogSoftmax.

    Returns:
        list: list of nn.Module, layers of the FCNN.
    """
    layers = []
    dim_in = in_dim
    for i, dim in enumerate(cfg):
        dim_out = dim[1] if isinstance(dim, tuple) else dim
        dim_in = dim[0] if isinstance(dim, tuple) else dim_in
        layers += [nn.Linear(dim_in, dim_out)]
        dim_in = dim_out
        if i < len(cfg)-1:
            layers += [nn.BatchNorm1d(dim_out), activ(),
                       nn.Dropout(p)] if bn else [activ(), nn.Dropout(p)]
            continue
    if out in (nn.Softmax, nn.LogSoftmax):
        layers += [out(-1)]  # explicit dimension choice for softmax
    else:
        layers += [out()]
    return layers

class Print(nn.Module):
    # debug 
    def __init__(self, txt=''):
        super(Print, self).__init__()
        self.txt = txt

    def forward(self, x):
        print(f'{self.txt}{x.shape}')
        return x

def make_conv_layers(cfg, in_channels=3, batch_norm=False, p=0.0, activation=nn.ReLU, pool_ks=2, pool_str=2, verbose=False):
    """Helper function to make layers of a Convolutionnal Neural Network. 

    Args:
        cfg (list): each element is either 'M' (pooling layer) or a tuple (out_channels, kernel_size, stride, padding) for a convolution layer
        in_channels (int, optional): input layer channels dimension. Defaults to 3.
        batch_norm (bool, optional): toggle Batch Normalization. Defaults to False.
        p (float, optional): Dropout parameters. Defaults to 0.0.
        activation (Type[nn.Module], optional): activation function constructor. Defaults to nn.ReLU.
        pool_ks (int, optional): kernel size for pooling layers. Defaults to 2.
        pool_str (int, optional): stride for pooling layers. Defaults to 2.

    Returns:
        list: list of nn.Module, layers of the CNN
    """
    layers = []
    current_in_channels = in_channels
    if verbose:
        layers += [Print('input')]
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=pool_ks, stride=pool_str)]
        else:
            ks = v[1] if isinstance(v, tuple) else 1
            stride = v[2] if isinstance(v, tuple) else 3
            pad = v[3] if isinstance(v, tuple) else 0
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(current_in_channels, out_channels,
                               kernel_size=ks, padding=pad, stride=stride)
            if batch_norm:
                layers += [conv2d,
                           nn.BatchNorm2d(out_channels), nn.Dropout(p), activation()]
            else:
                layers += [conv2d, nn.Dropout(p), activation()]
          
            current_in_channels = out_channels
        if verbose:
            layers += [Print('conv layer')]  
    return layers


def make_fc_layers_global_pooling(in_dim=512, out_shape=(9,9), num_classes=9):
    """Helper function to add a global pooling layer, useful for Fully Convolutional Network (no dense layers)
     to handle images of any sizes as input. 
    This trick, consists of successively applying a 1x1 Convolution, 
    and global pooling (e.g. adaptive average pooling). 

    Args:
        in_dim (int, optional): number of input channel, usually equivalent to output channels of the previous conv layer. Defaults to 512.
        out_shape (tuple[int,int], optional): desired output size for the DNN. Defaults to 9x9.
        num_classes(int, optional): number of classes. Defaults to 9.

    Returns:
        nn.ModuleList: 
    """
    # global pooling trick to handle any img input size
    class AveragePooling(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1x1 = torch.nn.Conv2d(in_channels=in_dim, out_channels=num_classes, kernel_size=1)
            self.global_pooling = torch.nn.AdaptiveAvgPool2d(out_shape)
        
        def forward(self, x):
            # B x nFeatureMaps x heightFeatureMaps x widthFeatureMaps
            h = self.global_pooling(x)
            # B x nFeatureMaps x heightShape x widthShape (out_dims)
            h = self.conv1x1(h)
            #h = torch.nn.Softmax(1)(h)
            # B x nClasses x out_dims
            bsize = len(h)
            # (needs to permute to B x out_dims x nClasses)
            out = h.permute(0, 2, 3, 1).contiguous().view(bsize, -1)
            return out.view(bsize, -1, num_classes)

    return [AveragePooling()]




def visu_sudoku(grid, figsize=(6,6)):
    N = int(math.sqrt(grid.shape[0]))

    # block-by-block alternation
    bg = np.zeros(grid.shape)
    for i in range(0,9, 3):
        for j in range(0,9, 3):
            if (i+j) % 2:
                bg[i:i+3, j:j+3] = 1
        
    # the figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.set(font_scale=2)
    sns.heatmap(bg, annot=grid,
                cbar=False, linewidths=1, xticklabels=False, yticklabels=False)
    
    plt.title(f"Sudoku {grid.shape[0]}x{grid.shape[1]}", fontsize=15)
    return ax

def get_sudoku_model(instance):
    n = len(instance)
    b = np.sqrt(n).astype(int)
    cells = cp.IntVar(1, n, shape=(n,n))

    # plain sudoku model
    m = cp.Model(
        [cp.alldifferent(row) for row in cells],
        [cp.alldifferent(col) for col in cells.T],
        [cp.alldifferent(cells[i:i + b, j:j + b])
            for i in range(0, n, b) for j in range(0, n, b)],
    )

    
    # set given clues
    m += cp.all(instance[instance>0] == cells[instance>0])
    return {
        'model':m,
        'variables':cells
    }

def solve_sudoku(model, dvars):
    if model.solve():
        results = {
            'runtime':np.asarray(model.cpm_status.runtime),
            'solution':dvars.value(),
        }
    else:
        results = {
            'solution':np.full_like(dvars.value(), np.nan)
        }
    results['status'] = model.cpm_status.exitstatus
    return results
