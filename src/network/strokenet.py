# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:09:19 2021

@author: BTLab
"""
import torch
import torch.nn as nn

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


class StrokeNet(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim=256, num_layers=2,
                 fc1_size=512, fc2_size=256, fc3_size=512, fc4_size=3,
                 pool='last'):
        super(StrokeNet, self).__init__()
        self.backbone = LSTM(input_dim, seq_len, hidden_dim, num_layers)
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
        lstm_out = self.backbone(x)
        pred = self.classifier(lstm_out)
        # Pred has shape (seq_len, batch_size, num_classes)
        if self.pool_type == 'last':
            pred = pred[-1]
        elif self.pool_type == 'max':
            pred = pred.transpose(0, 2)  # Tranpose first and third axes
            pred = pred.transpose(0, 1)  # Transpose first and second axes
            pred = self.pool(pred)
            pred = pred.squeeze()
        elif self.pool_type == 'avg':
            pred = pred.transpose(0, 2)  # Tranpose first and third axes
            pred = pred.transpose(0, 1)  # Transpose first and second axes
            pred = self.pool(pred)
            pred = pred.squeeze()
        elif self.pool_type == 'fc':
            pred = pred.transpose(0, 1)  # Transpose first and second axes
            pred = pred.reshape(pred.shape[0], -1)  # Vectorize each batch
            pred = self.pool(pred)  # Pass it through a fully connected layer
            pred = self.activation(pred)
        return pred


if __name__ == '__main__':
    # a = torch.randn(3, 15, 118*64)
    # net = LSTM(118*64, 15)
    # out = net(a)
    # clss = classification_head(out.shape[2])
    # pred = clss(out)
    import torch

    bar = torch.randn((8, 14400, 3))
    
    bar = bar.transpose(1,2)
    
    pool = torch.nn.MaxPool1d(3)
    
    if torch.cuda.is_available():
        bar.cuda()
        pool.cuda()
        
    
    spam = pool(bar)

