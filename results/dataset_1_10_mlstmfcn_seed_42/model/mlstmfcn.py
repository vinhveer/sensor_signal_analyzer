import torch
import torch.nn as nn
import torch.nn.functional as F
class SELayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _ = x.size()
        scale = self.avg_pool(x).view(batch_size, channels)
        scale = self.fc(scale).view(batch_size, channels, 1)
        return x * scale.expand_as(x)


class MLSTMfcn(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        max_seq_len: int,
        num_features: int,
        num_lstm_out: int = 128,
        num_lstm_layers: int = 1,
        conv1_nf: int = 128,
        conv2_nf: int = 256,
        conv3_nf: int = 128,
        lstm_drop_p: float = 0.2,
        fc_drop_p: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.num_features = num_features
        self.num_lstm_out = num_lstm_out
        self.num_lstm_layers = num_lstm_layers
        self.conv1_nf = conv1_nf
        self.conv2_nf = conv2_nf
        self.conv3_nf = conv3_nf
        self.lstm_drop_p = lstm_drop_p
        self.fc_drop_p = fc_drop_p

        self.lstm = nn.LSTM(
            input_size=self.num_features,
            hidden_size=self.num_lstm_out,
            num_layers=self.num_lstm_layers,
            batch_first=True,
        )
        self.conv1 = nn.Conv1d(self.num_features, self.conv1_nf, 8)
        self.conv2 = nn.Conv1d(self.conv1_nf, self.conv2_nf, 5)
        self.conv3 = nn.Conv1d(self.conv2_nf, self.conv3_nf, 3)
        self.bn1 = nn.BatchNorm1d(self.conv1_nf)
        self.bn2 = nn.BatchNorm1d(self.conv2_nf)
        self.bn3 = nn.BatchNorm1d(self.conv3_nf)
        self.se1 = SELayer(self.conv1_nf)
        self.se2 = SELayer(self.conv2_nf)
        self.se3 = SELayer(self.conv3_nf)
        self.relu = nn.ReLU()
        self.lstm_drop = nn.Dropout(self.lstm_drop_p)
        self.conv_drop = nn.Dropout(self.fc_drop_p)
        self.fc = nn.Linear(self.conv3_nf + self.num_lstm_out, self.num_classes)

    def forward(self, x: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        lstm_input = nn.utils.rnn.pack_padded_sequence(
            x,
            seq_lens,
            batch_first=True,
            enforce_sorted=False,
        )
        lstm_out, _ = self.lstm(lstm_input)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out,
            batch_first=True,
            padding_value=0.0,
        )
        lstm_out = self.lstm_drop(lstm_out[:, -1, :])

        conv_out = x.transpose(2, 1)
        conv_out = self.conv_drop(self.relu(self.bn1(self.conv1(conv_out))))
        conv_out = self.se1(conv_out)
        conv_out = self.conv_drop(self.relu(self.bn2(self.conv2(conv_out))))
        conv_out = self.se2(conv_out)
        conv_out = self.conv_drop(self.relu(self.bn3(self.conv3(conv_out))))
        conv_out = self.se3(conv_out)
        conv_out = torch.mean(conv_out, 2)

        features = torch.cat((lstm_out, conv_out), dim=1)
        return self.fc(features)


class MLSTMfcnClassifier(nn.Module):
    def __init__(self, in_channels: int =2, window: int =1024, num_classes: int =4):
        super().__init__()
        self.window = int(window)
        self.model = MLSTMfcn(
            num_classes=num_classes,
            max_seq_len=window,
            num_features=in_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_lens = torch.full((x.size(0),), self.window, dtype=torch.long)
        return self.model(x.transpose(1, 2), seq_lens)