"""
MobileNetV1 with width multiplier alpha=1.25
Output feature dimension: 1280  (matches fdim used in CATA-Net)

Usage:
    model = MobileNetV1(num_classes=52, alpha=1.25)
    logits = model(x)                        # (B, 52)
    feats  = model(x, return_features=True)  # (B, 1280)  for CATA-Net
"""

import torch
import torch.nn as nn


def _round_channels(c: float, divisor: int = 8) -> int:
    new_c = max(divisor, int(c + divisor / 2) // divisor * divisor)
    if new_c < 0.9 * c:
        new_c += divisor
    return new_c


class _DWSBlock(nn.Module):
    """Depthwise-separable conv block: DW BN ReLU6 → PW BN ReLU6."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class MobileNetV1(nn.Module):
    """
    MobileNetV1 (Howard et al. 2017) with configurable width multiplier.

    Default alpha=1.25 gives fdim=1280, matching the CATA-Net spec.
    """

    def __init__(self, num_classes: int = 52, alpha: float = 1.25):
        super().__init__()

        def ch(c: int) -> int:
            return _round_channels(c * alpha)

        self.features = nn.Sequential(
            # stem
            nn.Sequential(
                nn.Conv2d(3, ch(32), 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ch(32)),
                nn.ReLU6(inplace=True),
            ),
            # body
            _DWSBlock(ch(32),  ch(64),   stride=1),
            _DWSBlock(ch(64),  ch(128),  stride=2),
            _DWSBlock(ch(128), ch(128),  stride=1),
            _DWSBlock(ch(128), ch(256),  stride=2),
            _DWSBlock(ch(256), ch(256),  stride=1),
            _DWSBlock(ch(256), ch(512),  stride=2),
            # 5× repeated at 512
            _DWSBlock(ch(512), ch(512),  stride=1),
            _DWSBlock(ch(512), ch(512),  stride=1),
            _DWSBlock(ch(512), ch(512),  stride=1),
            _DWSBlock(ch(512), ch(512),  stride=1),
            _DWSBlock(ch(512), ch(512),  stride=1),
            # final downsample
            _DWSBlock(ch(512),  ch(1024), stride=2),
            _DWSBlock(ch(1024), ch(1024), stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fdim = ch(1024)       # 1280 for alpha=1.25
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.fdim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor,
                return_features: bool = False) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        feat = x.flatten(1)           # (B, fdim)
        if return_features:
            return feat
        return self.classifier(self.dropout(feat))


if __name__ == "__main__":
    model = MobileNetV1(num_classes=52, alpha=1.25)
    dummy = torch.randn(2, 3, 128, 128)
    print("logits shape :", model(dummy).shape)
    print("feature shape:", model(dummy, return_features=True).shape)
    print("fdim         :", model.fdim)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params       : {n_params/1e6:.2f} M")
