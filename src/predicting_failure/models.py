
import torch.nn as nn
import torch.nn.functional as F

class Recurrent(nn.Module):
    '''
    Model that will predict failure of teh jet engine using 
    '''
    def __init__(self, n_features=24, time_steps=150):
        super().__init__()

        # Defining the model layers
        self.n_features       = n_features
        self.inlayer          = nn.Linear(n_features, 64)
        self.lstm_layer       = nn.LSTM(input_size=n_features, hidden_size=128, num_layers=2, batch_first=True)
        self.dense_layer      = nn.Linear(128, 64)
        self.transition_layer = nn.Linear(64,  32)
        self.output_layer     = nn.Linear(32,  1)

        self.network_body = []



    def front_end(self):
        '''
        Front end of the model
        :return: Input layer
        '''
        # Input layer
        self.network_body.append(self.inlayer)
        self.network_body.append(nn.ReLU())
        # LSTM layer
        self.network_body.append(self.lstm_layer)
        self.network_body.append(nn.ReLU())

        return self.network_body


    def body(self):
        '''
        Will be a 32 input layer followed by an LSTM layer followed by dense layers 
        output layer will be a single value predicting remaining cycles
        '''

        for _ in range(4):
            self.network_body.append(self.dense_layer)
            self.network_body.append(nn.ReLU())

        return self.network_body


    def backend(self):
        '''
        Backend of the model
        :return: Output layer
        '''
        # Transition layer
        self.network_body.append(self.transition_layer)
        self.network_body.append(nn.ReLU())

        # Output layer
        self.network_body.append(self.output_layer)


        # Unpacking the model components
        model = nn.Sequential(*self.network_body)



        return model
    
    def construct_model(self):
        '''
        Constructs the model by combining front end, body, and backend
        :return: Complete model
        '''
        self.front_end()
        self.body()
        model = self.backend()
        return model
    

    def simple_model(self):
        '''
        Simple model construction
        '''

        model = nn.Sequential( self.inlayer,          
                               self.lstm_layer,
                               self.dense_layer,
                               self.transition_layer,
                               self.output_layer,
        ) 

        return model


    def forward(self, x):
        '''
        Forward pass of the model
        :param x: Input tensor
        :return: Output tensor
        '''
        # print(x.shape)
        # x = self.inlayer(x)          
        # x = x.permute(0, 2, 1)
        x, _= self.lstm_layer(x)
        x = nn.ReLU(x)
        x = self.dense_layer(x)
        x = nn.ReLU(x)
        x = self.transition_layer(x)
        x = nn.ReLU(x)
        x = self.output_layer(x)
        x = x.squeeze(-1) 
        
        return x 
    
    


class SingleRUL(nn.Module):
    '''
    Model that will predict failure of teh jet engine using 
    '''
    def __init__(self, n_features=24, time_steps=150):
        super().__init__()

        # Defining the model layers
        self.n_features       = n_features
        self.inlayer          = nn.Linear(n_features, 64)
        self.lstm_layer       = nn.LSTM(input_size=n_features, hidden_size=128, num_layers=2, batch_first=True)
        self.dense_layer      = nn.Linear(128, 64)
        self.transition_layer = nn.Linear(64,  32)
        self.output_layer     = nn.Linear(32,  1)

        self.network_body = []



    def forward(self, x):
        '''
        :param x: Input tensor of shape (batch, seq_len, features)
        :return: RUL predictions of shape (batch,)
        '''
        x, _ = self.lstm_layer(x)                  # (batch, seq_len, hidden)
        last_timestep = x[:, -1, :]                   # (batch, hidden)
        x = F.relu(self.dense_layer(last_timestep))
        x = F.relu(self.transition_layer(x))
        rul = self.output_layer(x)                 # (batch, 1)
        return rul.squeeze(-1)    
