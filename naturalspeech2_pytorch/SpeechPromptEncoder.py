import torch
import torch.nn as nn


class SpeechPromptEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        #Asuming input is 256 can change once we put all togather
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=2048, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=9, padding=4),
            nn.ReLU(),
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512, nhead=8, dim_feedforward=2048, dropout=0.2
            ),
            num_layers=6,
        )

    def forward(self, x):
        # x: (batch_size, seq_len, 256)
        x = x.transpose(1, 2)  # (batch_size, 256, seq_len)
        x = self.conv(x)  # (batch_size, 512, seq_len)
        x = x.transpose(0, 2)  # (seq_len, batch_size, 512)
        x = self.transformer(x)  # (seq_len, batch_size, 512)
        return x
