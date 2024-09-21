import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class biLSTM(torch.nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=6, dropout=0.2):
        super(biLSTM, self).__init__()
        
        # Bidirectional LSTM with increased hidden size and number of layers
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True  # Set to True for bidirectional LSTM
        )
        
        # Fully connected layer (output layer)
        self.fc = torch.nn.Linear(hidden_size * 2, 2)  # Multiply hidden size by 2 for bidirectional

        # Softmax layer for final classification
        self.softmax = torch.nn.Softmax(dim=1)
        
        # Load the pretrained model (optional)
        model = torch.load('results/pretrained-models/zv_bilstm_model.tar')
        my_dict = self.state_dict()
        for key, value in my_dict.items(): 
            my_dict[key] = model[key]
        self.load_state_dict(my_dict)
        self.eval()

    def forward(self, x, h=None):
        x = torch.FloatTensor(x).to(device).view((1, -1, 6))  # Input reshape

        if h is None:
            h_n = x.data.new(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).normal_(0, 0.1).to(device)  # 2 for bidirectional
            h_c = x.data.new(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).normal_(0, 0.1).to(device)
        else:
            h_n, h_c = h
        
        # LSTM forward pass
        self.lstm.flatten_parameters()
        r_out, (h_n, h_c) = self.lstm(x, (h_n, h_c))
        
        # Use fully connected layer to process output from LSTM (bidirectional)
        output = self.softmax(self.fc(r_out[0, :, :]))  # Output for the entire sequence

        # Post-processing to select the zero-velocity label
        zv_lstm = torch.max(output.cpu().data, 1)[1].numpy()
        prob = torch.max(output.cpu().data, 1)[0].numpy()
        zv_lstm[np.where(prob <= 0.85)] = 0  # Threshold to filter low-confidence predictions
        
        return zv_lstm  # Return predicted labels
