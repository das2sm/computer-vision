import torch
import torch.nn as nn
import torch.nn.functional as F

# Mish activation function
# Formula: x * tanh(softplus(x))
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
    

# Convolutional block 
# CBA pattern: Convolution + Batch Normalization + Activation
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation="mish"):
        super().__init__()
        # Use 'kernel_size // 2' padding to keep the image dimensions the same if stride=1
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Mish() if activation == "mish" else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    

# Residual Block
# Learn the residual mapping instead of the original mapping, which helps in training deeper networks
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Typically two layers: a 1x1 to reduce and a 3x3 to extract features
        self.conv1 = ConvBlock(channels, channels, kernel_size=1)
        self.conv2 = ConvBlock(channels, channels, kernel_size=3)

    def forward(self, x):
        # The Shortcut Connection (Residual)
        return x + self.conv2(self.conv1(x))


# Cross Stage Partial (CSP) Block
# This block splits the input into two parts, processes one part through a series of residual blocks
# and then merges it back with the other part, which helps in reducing computational cost while maintaining accuracy
class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks):
        super().__init__()
        half_channels = out_channels // 2
        
        # Branch 1
        self.skip_conv = ConvBlock(in_channels, half_channels, kernel_size=1)
        
        # Branch 2
        self.main_conv = ConvBlock(in_channels, half_channels, kernel_size=1)
        self.res_blocks = nn.Sequential(
            *[ResBlock(half_channels) for _ in range(num_res_blocks)]
        )
        self.transition_conv = ConvBlock(half_channels, half_channels, kernel_size=1)
        
        # Final Fusion
        self.fuse_conv = ConvBlock(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        part_a = self.skip_conv(x)
        part_b = self.transition_conv(self.res_blocks(self.main_conv(x)))
        return self.fuse_conv(torch.cat([part_a, part_b], dim=1))
    

class CSPDarknet53(nn.Module):
    def __init__(self):
        super().__init__()
        # Stem: 32 filters, 3x3 conv
        self.stem = ConvBlock(3, 32, kernel_size=3)
        
        # Stage 1: Downsample to 64, then CSP x 1
        self.stage1 = nn.Sequential(
            ConvBlock(32, 64, kernel_size=3, stride=2),
            CSPBlock(64, 64, num_res_blocks=1)
        )
        
        # Stage 2: Downsample to 128, then CSP x 2
        self.stage2 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=2),
            CSPBlock(128, 128, num_res_blocks=2)
        )
        
        # Stage 3: Downsample to 256, then CSP x 8 (Capture for Neck)
        self.stage3 = nn.Sequential(
            ConvBlock(128, 256, kernel_size=3, stride=2),
            CSPBlock(256, 256, num_res_blocks=8)
        )
        
        # Stage 4: Downsample to 512, then CSP x 8 (Capture for Neck)
        self.stage4 = nn.Sequential(
            ConvBlock(256, 512, kernel_size=3, stride=2),
            CSPBlock(512, 512, num_res_blocks=8)
        )
        
        # Stage 5: Downsample to 1024, then CSP x 4 (Capture for Neck)
        self.stage5 = nn.Sequential(
            ConvBlock(512, 1024, kernel_size=3, stride=2),
            CSPBlock(1024, 1024, num_res_blocks=4)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        
        out3 = self.stage3(x)  # Small features (52x52 for 416 input)
        out4 = self.stage4(out3) # Medium features (26x26)
        out5 = self.stage5(out4) # Large/Global features (13x13)
        
        return out3, out4, out5
    
    
if __name__ == "__main__":
    # GPU Verification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on: {device}")
    
    model = CSPDarknet53().to(device)
    dummy_input = torch.randn(1, 3, 416, 416).to(device)
    
    outputs = model(dummy_input)
    for i, out in enumerate(outputs):
        print(f"Stage {i+3} output shape: {out.shape}")