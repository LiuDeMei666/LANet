"""Common lightweight classifier heads shared across models."""

from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, flatten_number, n_classes, dropout=0.5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(flatten_number, n_classes),
        )

    def forward(self, x):
        return self.fc(x)


class GlobalAvgPoolClassificationHead(nn.Module):
    def __init__(self, embed_dim, n_classes, dropout=0.5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_classes),
        )

    def pool(self, token_features):
        return token_features.mean(dim=1)

    def forward(self, token_features):
        return self.fc(self.pool(token_features))
