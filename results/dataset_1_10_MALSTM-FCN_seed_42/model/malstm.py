import torch
import torch.nn as nn
import torch.nn.functional as F
class AttentionLSTM(nn.Module):
    """PyTorch equivalent of the supplied Keras AttentionLSTM."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = min(1.0, max(0.0, dropout))
        self.recurrent_dropout = min(1.0, max(0.0, recurrent_dropout))

        self.kernel = nn.Parameter(torch.empty(input_size, hidden_size * 4))
        self.recurrent_kernel = nn.Parameter(torch.empty(hidden_size, hidden_size * 4))
        self.attention_kernel = nn.Parameter(torch.empty(input_size, hidden_size * 4))
        self.attention_weights = nn.Parameter(torch.empty(input_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size * 4))
        self.attention_bias = nn.Parameter(torch.zeros(hidden_size))
        self.attention_recurrent_weights = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.attention_recurrent_bias = nn.Parameter(torch.empty(hidden_size, 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.kernel)
        nn.init.orthogonal_(self.recurrent_kernel)
        nn.init.orthogonal_(self.attention_kernel)
        nn.init.orthogonal_(self.attention_weights)
        nn.init.orthogonal_(self.attention_recurrent_weights)
        nn.init.orthogonal_(self.attention_recurrent_bias)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.attention_bias)

    @staticmethod
    def _hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(0.2 * x + 0.5, min=0.0, max=1.0)

    @staticmethod
    def _dropout_masks(
        batch_size: int,
        width: int,
        probability: float,
        reference: torch.Tensor,
    ) -> list[torch.Tensor]:
        if probability <= 0.0:
            return [reference.new_ones(batch_size, width) for _ in range(4)]
        if probability >= 1.0:
            return [reference.new_zeros(batch_size, width) for _ in range(4)]
        keep_probability = 1.0 - probability
        return [
            reference.new_empty(batch_size, width).bernoulli_(keep_probability) / keep_probability
            for _ in range(4)
        ]

    def forward(
        self,
        inputs: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.size(-1) != self.input_size:
            raise ValueError(
                f"Expected [batch, time, {self.input_size}], got {tuple(inputs.shape)}."
            )

        batch_size, timesteps, _ = inputs.shape
        hidden = inputs.new_zeros(batch_size, self.hidden_size)
        cell = inputs.new_zeros(batch_size, self.hidden_size)
        if self.training:
            input_masks = self._dropout_masks(
                batch_size, self.input_size, self.dropout, inputs
            )
            recurrent_masks = self._dropout_masks(
                batch_size, self.hidden_size, self.recurrent_dropout, inputs
            )
        else:
            input_masks = [inputs.new_ones(batch_size, self.input_size) for _ in range(4)]
            recurrent_masks = [inputs.new_ones(batch_size, self.hidden_size) for _ in range(4)]

        projected_attention = inputs @ self.attention_weights + self.attention_bias
        kernel_parts = self.kernel.chunk(4, dim=1)
        recurrent_parts = self.recurrent_kernel.chunk(4, dim=1)
        attention_parts = self.attention_kernel.chunk(4, dim=1)
        bias_parts = self.bias.chunk(4)
        outputs = []
        attention_outputs = []

        for timestep in range(timesteps):
            attention_energy = torch.tanh(
                (hidden @ self.attention_recurrent_weights).unsqueeze(1)
                + projected_attention
            )
            attention_logits = (attention_energy @ self.attention_recurrent_bias).squeeze(-1)
            attention = torch.softmax(attention_logits, dim=1)
            context_sequence = inputs * attention.unsqueeze(-1)
            context = context_sequence.sum(dim=1)

            gates = []
            current_input = inputs[:, timestep, :]
            for gate_index in range(4):
                gate = (
                    (current_input * input_masks[gate_index]) @ kernel_parts[gate_index]
                    + bias_parts[gate_index]
                    + (hidden * recurrent_masks[gate_index]) @ recurrent_parts[gate_index]
                    + context @ attention_parts[gate_index]
                )
                gates.append(gate)

            input_gate = self._hard_sigmoid(gates[0])
            forget_gate = self._hard_sigmoid(gates[1])
            candidate = torch.tanh(gates[2])
            output_gate = self._hard_sigmoid(gates[3])
            cell = forget_gate * cell + input_gate * candidate
            hidden = output_gate * torch.tanh(cell)
            outputs.append(hidden)
            if return_attention:
                attention_outputs.append(context_sequence)

        output_sequence = torch.stack(outputs, dim=1)
        if return_attention:
            return output_sequence, torch.stack(attention_outputs, dim=1)
        return output_sequence


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
        lstm_drop_p: float = 0.1,
        fc_drop_p: float = 0.1,
        attention_pool_stride: int = 4,
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
        self.attention_pool_stride = attention_pool_stride

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
        self.attention_pool1 = nn.MaxPool1d(
            kernel_size=self.attention_pool_stride,
            stride=self.attention_pool_stride,
        )
        self.attention_pool2 = nn.MaxPool1d(
            kernel_size=self.attention_pool_stride,
            stride=self.attention_pool_stride,
        )
        self.attention_pool3 = nn.MaxPool1d(
            kernel_size=self.attention_pool_stride,
            stride=self.attention_pool_stride,
        )
        self.attention1 = AttentionLSTM(self.conv1_nf, self.conv1_nf)
        self.attention2 = AttentionLSTM(self.conv2_nf, self.conv2_nf)
        self.attention3 = AttentionLSTM(self.conv3_nf, self.conv3_nf)
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
        conv_out = self.attention_pool1(conv_out)
        conv_out = self.attention1(conv_out.transpose(1, 2)).transpose(1, 2)
        conv_out = self.conv_drop(self.relu(self.bn2(self.conv2(conv_out))))
        conv_out = self.attention_pool2(conv_out)
        conv_out = self.attention2(conv_out.transpose(1, 2)).transpose(1, 2)
        conv_out = self.conv_drop(self.relu(self.bn3(self.conv3(conv_out))))
        conv_out = self.attention_pool3(conv_out)
        conv_out = self.attention3(conv_out.transpose(1, 2)).transpose(1, 2)
        conv_out = torch.mean(conv_out, 2)

        features = torch.cat((lstm_out, conv_out), dim=1)
        return self.fc(features)


class MLSTMfcnClassifier(nn.Module):
    def __init__(self, in_channels: int, window: int, num_classes: int):
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