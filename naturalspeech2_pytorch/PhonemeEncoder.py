import torch
import torch.nn as nn

class PhonemeEncoder(nn.Module):
    def __init__(self, num_layers=6, num_heads=8, hidden_size=512, filter_size=2048, kernel_size=9, dropout=0.2):
        super(PhonemeEncoder, self).__init__()

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=filter_size, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=filter_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # input would be text_to_ids output
        x = x.transpose(1, 2)  # (B, H, S) -> (B, S, H)
        x = self.conv1d(x)  # (B, S, H) -> (B, F, S)
        x = x.transpose(0, 1)  # (B, F, S) -> (S, B, F)

        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Final output shape: (S, B, F)
        return x.transpose(0, 1)  # (S, B, F) -> (B, S, F)