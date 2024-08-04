# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

import numpy as np
import torch
from torch import nn
from torch.functional import F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Network(nn.Module):
    def __init__(self, seq_len, input_size = 3):
        super(Network, self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.input_size, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.hidden_size = 128
        self.num_layers = 2
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first=True, nonlinearity='relu', dropout=0.3)
        self.fc2 = nn.Linear(self.hidden_size, 3)
    
    def forward(self, x):
        x = x.reshape(1,seq_len,3)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        x = self.fc(x.reshape(1,-1)) + self.fc2(out)
        return x
    

one_hot = {
    'R': torch.tensor([[1., 0., 0.]]).to(device),
    'P': torch.tensor([[0., 1., 0.]]).to(device),
    'S': torch.tensor([[0., 0., 1.]]).to(device)
}

counter_move = {
    'R': 'P',
    'P': 'S',
    'S': 'R'
}

seq_len = 6
    



def player(prev_play, opponent_history=[]):
    global model, loss_fn, optimizer
    guess = np.random.choice(['R', 'P', 'S'])

    if prev_play == '':
        model = Network(seq_len).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
        return guess
    
    opponent_history.append(prev_play)

    if len(opponent_history) > 1 + seq_len:
        model.train()
        guess = model(torch.concat([one_hot[move] for move in opponent_history[-1 - seq_len:-1]]))
        pred = one_hot[counter_move[prev_play]].to(device).argmax(dim=1)
        loss = loss_fn(guess, pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        with torch.inference_mode():
            model.eval()
            guess = list(one_hot)[model(torch.concat([one_hot[move] for move in opponent_history[-seq_len:]])).argmax(dim=1).item()]
            
    return guess
