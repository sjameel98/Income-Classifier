import torch.nn as nn


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size, hidden_layers, activation):

        super(MultiLayerPerceptron, self).__init__()

        ######

        # 3.3 YOUR CODE HERE
        self.input_size = input_size
        self.output_size = 1
        if (activation == 'Tanh'):
            self.activation = nn.Tanh()
        elif (activation == 'Sigmoid'):
            self.activation = nn.Sigmoid()
        elif (activation == 'ReLU'):
            self.activation = nn.ReLU()
        if (hidden_layers == None):
            self.hidden_layer_number = 0
        else:
            self.hidden_layer_number = len(hidden_layers)
        if self.hidden_layer_number == 0:
            self.fc1 = nn.Linear(input_size, self.output_size)
        elif self.hidden_layer_number == 1:
            self.fc1 = nn.Linear(input_size, hidden_layers[0])
            self.fc2 = nn.Linear(hidden_layers[0], self.output_size)
        elif self.hidden_layer_number == 2:
            self.fc1 = nn.Linear(input_size, hidden_layers[0])
            self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
            self.fc3 = nn.Linear(hidden_layers[1],self.output_size)
        elif  self.hidden_layer_number == 3:
            self.fc1 = nn.Linear(input_size, hidden_layers[0])
            #print(self.fc1)
            self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
            #print(self.fc2)
            self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])
            #print(self.fc3)
            self.fc4 = nn.Linear(hidden_layers[2], self.output_size)
            #print(self.fc4)
    def forward(self, features):

        pass
        ######

        # 3.3 YOUR CODE HERE
        x = self.fc1(features)
        #print(features.shape)

        if self.hidden_layer_number >= 1:
            x = self.activation(x)
            x= self.fc2(x)
        if self.hidden_layer_number >= 2:
            x = self.activation(x)
            x = self.fc3(x)
            #print(x.shape)
        if self.hidden_layer_number >= 3:
            x = self.activation(x)
            x = self.fc4(x)
        act2 = nn.Sigmoid()
        x = act2(x)
        return x

        ######