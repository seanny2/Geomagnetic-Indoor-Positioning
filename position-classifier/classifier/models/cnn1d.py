import torch.nn as nn

class PositionClassifier(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PositionClassifier, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=in_channel,
                                out_channels=16,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=16,
                                out_channels=32,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.conv1d_3 = nn.Conv1d(in_channels=32,
                                out_channels=64,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.conv1d_4 = nn.Conv1d(in_channels=64,
                                out_channels=128,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=50,
                            num_layers=1,
                            bias=True,
                            bidirectional=False,
                            batch_first=True)
        
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Sequential(
            nn.Linear(50, out_channel),
            nn.ReLU(),
            nn.BatchNorm1d(out_channel),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        x = self.conv1d_4(x)
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)
        x = hidden[-1]
        x = self.dropout(x)
        x = self.fc(x)

        return x