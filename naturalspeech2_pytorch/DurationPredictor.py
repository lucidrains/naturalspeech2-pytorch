import torch
import torch.nn as nn

class DurationPredictor(nn.Module):
    def __init__(self, n_layers=30, kernel_size=3, n_att_layers=10, n_heads=8, hidden_size=512, dropout=0.2):
        super(DurationPredictor, self).__init__()

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2)
            for _ in range(n_layers)
        ])

        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_size, n_heads)
            for _ in range(n_att_layers)
        ])

        self.fc = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, prompt_encoder_output):
        # x shape: (batch_size, hidden_size, length)
        # prompt_encoder_output shape: (prompt_length, batch_size, hidden_size)

        # Transpose prompt_encoder_output to (batch_size, hidden_size, prompt_length)
        prompt_encoder_output = prompt_encoder_output.transpose(0, 1)

        # Apply Q-K-V attention layer
        for att_layer in self.attention_layers:
            x, _ = att_layer(x, prompt_encoder_output, prompt_encoder_output)

        # Apply conv layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = torch.relu(x)
            x = self.dropout(x)

        # Apply fully connected layer
        x = self.fc(x.transpose(1, 2)).squeeze(-1)

        # Make sure predicted durations are positive
        x = torch.relu(x)

        return x
