import torch
import torch.nn as nn
import torch.fft
from torchvision import models

class FFTResNet(nn.Module):
    def __init__(self, num_classes, pretrained=True, freeze_backbone=False):
        super().__init__()
        
        # Load base resnet
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # 1. Modify the first convolutional layer to accept 6 channels (3 RGB + 3 FFT)
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=6, 
            out_channels=old_conv.out_channels, 
            kernel_size=old_conv.kernel_size, 
            stride=old_conv.stride, 
            padding=old_conv.padding, 
            bias=old_conv.bias is not None
        )
        
        # 2. Initialize the new conv layer 
        if pretrained:
            # Copy original pre-trained weights for the first 3 channels (RGB)
            self.backbone.conv1.weight.data[:, :3, :, :] = old_conv.weight.data
            
            # Initialize the next 3 channels (FFT) with scaled down versions of the original weights
            # This helps the network learn the new FFT features without destabilizing early training
            self.backbone.conv1.weight.data[:, 3:, :, :] = old_conv.weight.data * 0.1

        # 3. Replace the final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        # 4. Handle freezing
        if freeze_backbone:
            for name, p in self.backbone.named_parameters():
                # We must NEVER freeze conv1 now, because half of its weights are new!
                if not name.startswith('fc') and not name.startswith('conv1'):
                    p.requires_grad = False

    def forward(self, x):
        # x shape: [Batch, 3, Height, Width]
        
        # Compute 2D Fast Fourier Transform
        fft_x = torch.fft.fft2(x)
        
        # Shift the zero-frequency component to the center of the spectrum
        fft_shifted = torch.fft.fftshift(fft_x, dim=(-2, -1))
        
        # Get magnitude (absolute value)
        fft_mag = torch.abs(fft_shifted)
        
        # Log scaling: log(1 + x). 
        # Crucial because FFT center frequencies are exponentially larger than high frequencies.
        fft_mag = torch.log1p(fft_mag)
        
        # Concatenate RGB and FFT along the channel dimension
        # combined shape: [Batch, 6, Height, Width]
        combined = torch.cat([x, fft_mag], dim=1)
        
        # Pass through the modified ResNet
        return self.backbone(combined)


def build_model(num_classes, cfg=None):
    cfg = cfg or {}
    pretrained = cfg.get("pretrained", True)
    freeze_backbone = cfg.get("freeze_backbone", False)

    # Return our custom model instead of the default ResNet
    model = FFTResNet(num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    
    return model


if __name__ == '__main__':
    # quick smoke test
    m = build_model(10)
    # create a dummy image (Batch size 2, 3 channels, 224x224)
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    print(f"Output shape: {y.shape}") # Should be [2, 10]