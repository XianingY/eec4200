from __future__ import annotations


def _require_torch():
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for training and evaluation. Install dependencies with "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc
    return nn


class Conv3DBlock:
    """Factory wrapper so the module can be imported before PyTorch is installed."""

    def __new__(cls, in_channels: int, out_channels: int, pool_kernel=None):
        nn = _require_torch()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool_kernel is not None:
            layers.append(nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_kernel))
        return nn.Sequential(*layers)


class Lightweight3DCNN:
    def __new__(cls, num_classes: int = 8, dropout: float = 0.4):
        nn = _require_torch()

        class _Lightweight3DCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    Conv3DBlock(3, 32, pool_kernel=(1, 2, 2)),
                    Conv3DBlock(32, 64, pool_kernel=(2, 2, 2)),
                    Conv3DBlock(64, 128, pool_kernel=(2, 2, 2)),
                    Conv3DBlock(128, 256, pool_kernel=None),
                )
                self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
                self.dropout = nn.Dropout(p=dropout)
                self.classifier = nn.Linear(256, num_classes)

            def forward(self, inputs):
                features = self.features(inputs)
                pooled = self.pool(features)
                flattened = pooled.flatten(1)
                return self.classifier(self.dropout(flattened))

        return _Lightweight3DCNN()


def architecture_table() -> list[dict[str, object]]:
    return [
        {"stage": "Input", "shape": "3 x 16 x 112 x 112", "details": "RGB clip with 16 frames"},
        {"stage": "Block 1", "shape": "32 x 16 x 56 x 56", "details": "Conv3d + BN + ReLU + MaxPool3d(1,2,2)"},
        {"stage": "Block 2", "shape": "64 x 8 x 28 x 28", "details": "Conv3d + BN + ReLU + MaxPool3d(2,2,2)"},
        {"stage": "Block 3", "shape": "128 x 4 x 14 x 14", "details": "Conv3d + BN + ReLU + MaxPool3d(2,2,2)"},
        {"stage": "Block 4", "shape": "256 x 4 x 14 x 14", "details": "Conv3d + BN + ReLU"},
        {"stage": "Head", "shape": "8", "details": "Global average pooling + Dropout(0.4) + Linear"},
    ]
