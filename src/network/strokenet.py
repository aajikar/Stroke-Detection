# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:09:19 2021

@author: BTLab
"""
import torch
import torch.nn as nn
from functools import reduce
import functools
import time

# The input of the incoming data will be n, t where n is the vector of x, y
# pressure data and t is the number of frames in time

# First create a input class layer of lstm layers
# Takes in shape (118x64, t)
# t should be user defined if possible


class LSTM(nn.Module):
    """LSTM class that takes in the initial input."""

    def __init__(self, input_dim, seq_len, hidden_dim=256, num_layers=2):
        """
        RNN portion of the nn using LSTM layers.

        Parameters
        ----------
        input_shape : int
            The number of dimensions in a pressure frame. This is a vector of
            size rows*cols.
        hidden_dim : int, optional
            The size of the hidden dimension. The default is 256.
        num_layers : int, optional
            Number of LSTM layers. The default is 2.

        Returns
        -------
        None.

        """
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        # Create the LSTM layer
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=0.5)

    def reset_hidden_state(self):
        """
        Reset the hidden state back to zero.

        The hidden layer by default is put on the GPU.

        Returns
        -------
        None.

        """
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len, self.hidden_dim).cuda(),
            torch.zeros(self.n_layers, self.seq_len, self.hidden_dim).cuda()
            )

    def forward(self, x):
        """
        Forward method of the model.

        The forward method uses lstm which resets the hidden state after every
        run. This is because previous sequence is not important for the
        prediction of the next sequence.

        Parameters
        ----------
        x : torch.Tensor
            Tensor in the shape (batch_size, seq_len, 5664).
            Where 5664 represents the vectorized pressure image of size 118*64.

        Returns
        -------
        out : torch.Tensor
            Output tensor in the shape (seq_len, batch_size, hidden_dim). This
            is the standard ouput that is ready to be fed into fully connected
            layers.

        """
        batch_size = len(x)
        
        # Reshape the input
        x = x.view(batch_size, self.seq_len, -1)
        x = x.cuda()

        # Hidden state
        self.reset_hidden_state()

        # Passing in the input and hidden state into model
        lstm_out, _ = self.lstm(x, self.hidden)
        
        # Reshape the output so it can be fed into fully connected layer
        out = lstm_out.view(self.seq_len, batch_size, self.hidden_dim)
        
        return out
    
    def init_hidden(self, batch_size):
        """
        Intialize the hidden layer of the LSTM.

        Parameters
        ----------
        batch_size : int
            Integer number representing the batch size.

        Returns
        -------
        hidden : torch.Tensor
            Hidden layer of the shape (n_layers, batch_size, hidden_dim).
            The tensor is all zeros.

        """
        # This method generates the first hidden state of zeros for forwad pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


class classification_head(nn.Module):
    """
    Class containing sequential fully connected layers.

    The class has four fully connected (fc) layers defualt to the size of
    512-256-512-3. The first three fc layers are followed by a ReLu function
    and a dropout of 0.5.

    The final fc layer has a following Softmax layer.
    """

    def __init__(self, hidden_dim, fc1_size=512, fc2_size=256, fc3_size=512,
                 fc4_size=3):
        """
        Initialize classfication head with four fully connected layers.

        The input size of the first layer is the hidden dimension of LSTM.

        Parameters
        ----------
        hidden_dim : int
            The input size of the first fully connected layer. For LSTMs this
            will be the hidden dimension.
        fc1_size : int, optional
            The size of the first hidden layer. The default is 512.
        fc2_size : int, optional
            The size of the second hidden layer. The default is 256.
        fc3_size : int, optional
            The size of the third hidden layer. The default is 512.
        fc4_size : int, optional
            The size of the fourth hidden layer. The default is 3.

        Returns
        -------
        None.

        """
        super(classification_head, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.fc4 = nn.Linear(fc3_size, fc4_size)

        self.d1 = nn.Dropout(p=0.5)
        self.d2 = nn.Dropout(p=0.5)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.activation = nn.Softmax(dim=2)

    def forward(self, x):
        """
        Forward method of the classification head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (seq_len, batch_size, hidden_dim). The hidden
            dim is the size of the LSTM hidden_dim variable.

        Returns
        -------
        x : torch.Tensor
            Output tensor from the shape (seq_len, batch_size, num_classes).

        """
        x = self.relu1(self.fc1(x))
        x = self.d1(x)
        x = self.relu2(self.fc2(x))
        x = self.d2(x)
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return self.activation(x)


class backbone(nn.Module):
    """Class for testing out different sizes of backbones."""

    def __init__(self, num_conv_layers, *sizes, pool_type='max',
                 pool_size=(12,4)):
        """
        Initialize the backbone with specified number of layers and sizes.

        The netowrk consists of a convolution backbone that is user defined.
        The backbone can be any integer number of layers deep. The sizes are
        also user defined.

        At the end of the convolutional layers is an adaptive pooling layer.
        The pooling type 'avg' or 'max' can be user defined. The pool size can
        also be user defined.

        Parameters
        ----------
        num_conv_layers : int
            The number of convolutional layers in the backbone. The first
            convolution layer is of size 7x7 with a stride of 2. The following
            convolution layers have kernel size of 3x3 and stride of 1.
        *sizes : int
            Output sizes of the convolutional layers. The number of sizes must
            be equal to the number of layers. If the sizes are not defined the
            default sizes will be 512 for odd numbered layers and 256 for even
            numbered layers.
        pool_type : str, optional
            The type of pooling that should occur after the convolutions. There
            are two choices 'max' or 'avg'. If another value is provided an
            error will be raised. The default is 'max'.
        pool_size : tuple, optional
            The size of the pooling layer. Since a 2D pooling is applied the
            size must be a tuple of two numbers. The first number is the height
            and the second number is the width. If only one number is given the
            pool size will be of a square shape. The default is (12,4).

        Raises
        ------
        ValueError
            A value error is raised when something other than 'max' or 'avg' is
            provided as the pool_type.

        Returns
        -------
        None.

        """
        super(backbone, self).__init__()
        # If sizes are not defined, set default
        if len(sizes) == 0:
            self.sizes = []
            for i in range(num_conv_layers):
                if (i == 0) or (i % 2 == 0):
                    self.sizes.append(512)
                else:
                    self.sizes.append(256)
        else:
            # First assert the sizes match with num_conv_layers
            assert num_conv_layers == len(sizes), "The number of sizes should match the number of layers"
            self.sizes = sizes
        self.num_layers = num_conv_layers
        
        conv_layers = {}
        for i in range(self.num_layers):
            if i == 0:
                conv_layers["conv{0}".format(i)] = nn.Conv2d(1, self.sizes[i], 
                                                             7, stride=2).cuda()
            else:
                conv_layers["conv{0}".format(i)] = nn.Conv2d(self.sizes[i-1],
                                                             self.sizes[i], 3).cuda()
        
        self.conv_layers = conv_layers
        
        self.relu_layers = {}
        for i in range(self.num_layers):
            self.relu_layers['relu{0}'.format(i)] = nn.ReLU().cuda()
        
        self.batchnorm_layers = {}
        for i in range(self.num_layers):
            self.batchnorm_layers['bn{0}'.format(i)] = nn.BatchNorm2d(self.sizes[i]).cuda()


        self.layers = {'conv': self.conv_layers,
                       'relu': self.relu_layers,
                       'bn': self.batchnorm_layers}
        

        self.keys = list(self.layers.keys())
        
        self.pool_size = pool_size
        
        if pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(pool_size).cuda()
        elif pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(pool_size).cuda()
        else:
            raise ValueError("The pool type must be avg or max")

    def forward(self, x):
        """
        Forward method of the backbone module.

        The incoming tensor is unsequeezed in dim=2. Then the tensor is
        reshpaed such to form (batch*seq_len, C, H, W). Where batch is the size
        of the batch, seq_len is the length of the sequence, C is the number of
        channels which is the unsqueezed dimension, H is the height, and W is
        the width of the pressure frame. Once the pooling has been perfromed,
        the tensor is reshaped to shape (batch, seq_len, C_new, H_new, W_new).

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the model. The shape of the tensor should be in
            the form (batch, sequence length, H, W).

        Returns
        -------
        x : torch.Tensor
            The output tesnor from the model. The shape of the tensor is in the
            shape (batch, sequence length, C_new, H_new, W_new). H_new and
            W_new correspond to the pool size. C_new is the number of feature
            channels from the last convolution layer.

        """
        # The incoming tensor is of shape (batch_size, seq_len, H, W)
        # First unsqueeze the tensor so shape is (batch, seq_len, C, H, W)
        # Where C = 1
        # Then view the tensor as (batch*seq_len, C, H, W)
        # Pass it through the CNN
        # This will result in the shape (batch*seq_len, C_new, H_new, W_new)
        # View the tensor as (batch, seq_len, -1)
        # The shape will be (batch, seq_len, C_new, H_new, W_new)
        
        x = x.unsqueeze(2)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size * seq_len, x.shape[2], x.shape[3], x.shape[4])
        
        for j in range(self.num_layers):
            for key in self.keys:
                x = self.layers[key][key + str(j)](x)
        
        x = self.pool(x)
        
        x = x.view(batch_size, seq_len, -1)
        
        return x


class StrokeNet(nn.Module):
    """First version of StrokeNet"""

    def __init__(self, num_conv_layers=3, *sizes, conv_pool_type='max',
                 pool_size=(12,4), seq_len=600, hidden_dim=256, num_layers=2,
                 fc1_size=512, fc2_size=256, fc3_size=512, fc4_size=3,
                 pool='last', input_dim=None, binary=False):

        super(StrokeNet, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.sizes = sizes
        self.conv_pool_type = conv_pool_type
        
        if self.num_conv_layers == 0:
            assert input_dim is not None, "input_dim must be defined when num_conv_layers is 0"
            self.input_dim = input_dim
        else:
            self.backbone = backbone(num_conv_layers, *sizes, pool_type=conv_pool_type,
                                     pool_size=pool_size)
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
            self.backbone.to(device)
            if len(self.sizes) == 0:
                # 512 is default size of last conv if sizes is not defined
                input_dim = pool_size[0] * pool_size[1] * 512
            else:
                input_dim = pool_size[0] * pool_size[1] * sizes[-1]

        if binary:
            fc4_size = 2

        self.lstm = LSTM(input_dim, seq_len, hidden_dim, num_layers)
        self.classifier = classification_head(hidden_dim, fc1_size, fc2_size,
                                              fc3_size, fc4_size)
        self.pool_type = pool
        if pool == 'max':
            self.pool = nn.MaxPool1d(seq_len)
        elif pool == 'avg':
            self.pool = nn.AvgPool1d(seq_len)
        elif pool == 'fc':
            self.pool = nn.Linear(seq_len*fc4_size, fc4_size)
            self.activation = nn.Softmax(dim=1)
        else:
            self.pool = None


    def forward(self, x):
        if self.num_conv_layers:
            x = self.backbone(x)
        x = self.lstm(x)
        x = self.classifier(x)
        # Pred has shape (seq_len, batch_size, num_classes)
        if self.pool_type == 'last':
            x = x[-1]
        elif self.pool_type == 'max':
            x = x.transpose(0, 2)  # Tranpose first and third axes
            x = x.transpose(0, 1)  # Transpose first and second axes
            x = self.pool(x)
            x = x.squeeze()
        elif self.pool_type == 'avg':
            x = x.transpose(0, 2)  # Tranpose first and third axes
            x = x.transpose(0, 1)  # Transpose first and second axes
            x = self.pool(x)
            x = x.squeeze()
        elif self.pool_type == 'fc':
            x = x.transpose(0, 1)  # Transpose first and second axes
            x = x.reshape(x.shape[0], -1)  # Vectorize each batch
            x = self.pool(x)  # Pass it through a fully connected layer
            x = self.activation(x)
        return x


class BodiesAtRestBackbone(nn.Module):
    def __init__(self, pool_type='max'):
        super(BodiesAtRestBackbone, self).__init__()
        self.conv1 = nn.Conv2d(1, 192, kernel_size=7, stride=2)
        self.conv2 = nn.Conv2d(192, 192, 3)
        self.conv3 = nn.Conv2d(192, 384, 3)
        self.conv4 = nn.Conv2d(384, 192, 3)
        
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.tanh3 = nn.Tanh()
        self.tanh4 = nn.Tanh()
        
        self.bn1 = nn.BatchNorm2d(192)
        self.bn2 = nn.BatchNorm2d(192)
        self.bn3 = nn.BatchNorm2d(384)
        self.bn4 = nn.BatchNorm2d(192)
        
        self.pool_type = pool_type
        if self.pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d((10, 4))
        elif self.pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((10, 4))
        else:
            raise ValueError('The pool_type must be avg or max')
    
    def forward(self, x):
        x = x.unsqueeze(2)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size * seq_len, x.shape[2], x.shape[3], x.shape[4])
        
        x = self.conv1(x)
        x = self.tanh1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.tanh3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.tanh4(x)
        x = self.bn4(x)
        x = self.pool(x)
        
        x = x.view(batch_size, seq_len, -1)
        
        return x


class StrokeNetV2(nn.Module):
    def __init__(self, seq_len):
        super(StrokeNetV2, self).__init__()
        self.backbone = BodiesAtRestBackbone()
        self.lstm = LSTM(10*4*192, seq_len)
        self.classifier = classification_head(256, fc4_size=2)
        self.pool = nn.Linear(3*seq_len, 3)
        self.activation = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.lstm(x)
        x = self.classifier(x)
        x = x.transpose(0, 1)  # Transpose first and second axes
        x = x.reshape(x.shape[0], -1)  # Vectorize each batch
        x = self.pool(x)
        x = self.activation(x)
        return x

# if __name__ == '__main__':
    # a = torch.randn(3, 15, 118*64)
    # net = LSTM(118*64, 15)
    # out = net(a)
    # clss = classification_head(out.shape[2])
    # pred = clss(out)
    # import torch

    # bar = torch.randn((8, 14400, 3))
    
    # bar = bar.transpose(1,2)
    
    # pool = torch.nn.MaxPool1d(3)
    
    # m = StrokeNet(seq_len=6)
        
    # foo = torch.randn((2, 6, 118, 48))
    
    # bar = m.forward(foo)
    
    # def timer(func):
    #     """Print the runtime of the decorated function."""
    #     @functools.wraps(func)
    #     def wrapper_timer(*args, **kwargs):
    #         start_time = time.perf_counter_ns()
    #         value = func(*args, **kwargs)
    #         end_time = time.perf_counter_ns()
    #         run_time = start_time - end_time
    #         print(f"Finished {func.__name__!r} in {run_time:.4f} ns")
    #         return value
    #     return wrapper_timer
    
    # @timer
    # def l_method(tup):
    #     print(reduce(lambda x, y : x*y, tup))
    
    # @timer
    # def s_method(tup):
    #     print(tup[0] * tup[1])
    
    # tup = (12, 4)
    
    # l_method(tup)
    
    # s_method(tup)
