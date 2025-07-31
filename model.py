import torch
import torch.nn as nn

class OptimizedGRUModel(nn.Module):
    """Định nghĩa kiến trúc mô hình GRU."""
    def __init__(self, input_size, num_classes, hidden_size=64, dropout=0.2):
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size, bias=False)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes, bias=False)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                if "gru" in name:
                    nn.init.orthogonal_(param, gain=0.8)
                else:
                    nn.init.xavier_uniform_(param, gain=0.6)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_projection(x)
        gru_out, _ = self.gru(x)
        x = self.dropout(gru_out[:, -1, :])
        return self.classifier(x)
