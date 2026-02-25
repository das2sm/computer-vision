import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Backbone ---

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
    

# Complete backbone architecture based on CSPDarknet53
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
    

# --- 2. Neck ---

# A sequence of 5 alternating 1x1 and 3x3 convolutions
# Used for feature refinement and fusion in the neck.
class FiveConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        inner_channels = out_channels * 2 # Standard YOLOv4 pattern
        self.conv = nn.Sequential(
            ConvBlock(in_channels, out_channels, 1),
            ConvBlock(out_channels, inner_channels, 3),
            ConvBlock(inner_channels, out_channels, 1),
            ConvBlock(out_channels, inner_channels, 3),
            ConvBlock(inner_channels, out_channels, 1),
        )

    def forward(self, x):
        return self.conv(x)


# Spatial pyramid pooling (SPP) module
# This module applies max pooling with different kernel sizes to capture features at multiple scales.
class SPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cv1 = ConvBlock(in_channels, out_channels, 1)
        self.m5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.m9 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.m13 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)

    def forward(self, x):
        x = self.cv1(x)
        return torch.cat([self.m13(x), self.m9(x), self.m5(x), x], dim=1)


# Neck architecture that implements the PANet structure for feature fusion
class YOLOv4Neck(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- TOP-DOWN PATH ---
        # 1. SPP on the 13x13 (Stage 5)
        self.spp = SPP(1024, 512)
        self.cv_spp = ConvBlock(2048, 512, 1) # Reduce channels after SPP cat
        
        # 2. Transition Stage 5 -> Stage 4 (Upsample)
        self.cv5 = ConvBlock(512, 256, 1)
        self.upsample5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv4_map = ConvBlock(512, 256, 1) # Maps stage4 output to match
        self.fusion54 = FiveConvBlock(512, 256)
        
        # 3. Transition Stage 4 -> Stage 3 (Upsample)
        self.cv4 = ConvBlock(256, 128, 1)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv3_map = ConvBlock(256, 128, 1) # Maps stage3 output to match
        self.fusion43 = FiveConvBlock(256, 128)

        # --- BOTTOM-UP PATH ---
        # 4. Transition Stage 3 -> Stage 4 (Downsample)
        self.down3 = ConvBlock(128, 256, 3, stride=2)
        self.fusion34 = FiveConvBlock(512, 256) # 256(down) + 256(prev)
        
        # 5. Transition Stage 4 -> Stage 5 (Downsample)
        self.down4 = ConvBlock(256, 512, 3, stride=2)
        self.fusion45 = FiveConvBlock(1024, 512) # 512(down) + 512(prev)

    def forward(self, s3, s4, s5):
        # Top-Down
        p5 = self.cv_spp(self.spp(s5))
        
        p5_up = self.upsample5(self.cv5(p5))
        p4 = torch.cat([self.cv4_map(s4), p5_up], dim=1)
        p4 = self.fusion54(p4)
        
        p4_up = self.upsample4(self.cv4(p4))
        p3 = torch.cat([self.cv3_map(s3), p4_up], dim=1)
        p3 = self.fusion43(p3) # FIRST OUTPUT (Small objects)
        
        # Bottom-Up
        p3_down = self.down3(p3)
        p4_fuse = torch.cat([p3_down, p4], dim=1)
        p4_final = self.fusion34(p4_fuse) # SECOND OUTPUT (Medium objects)
        
        p4_down = self.down4(p4_final)
        p5_fuse = torch.cat([p4_down, p5], dim=1)
        p5_final = self.fusion45(p5_fuse) # THIRD OUTPUT (Large objects)
        
        return p3, p4_final, p5_final
    

class YOLOv4(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.backbone = CSPDarknet53()
        self.neck = YOLOv4Neck()
        
        # Heads: Final 1x1 convs to get the correct number of classes/anchors
        # Each anchor predicts: [x, y, w, h, confidence, ...classes]
        # (Assuming 3 anchors per scale)
        out_channels = 3 * (5 + num_classes)
        self.head3 = nn.Conv2d(128, out_channels, 1)
        self.head4 = nn.Conv2d(256, out_channels, 1)
        self.head5 = nn.Conv2d(512, out_channels, 1)

    def forward(self, x):
        s3, s4, s5 = self.backbone(x)
        p3, p4, p5 = self.neck(s3, s4, s5)
        
        return self.head3(p3), self.head4(p4), self.head5(p5)
    

if __name__ == "__main__":
    # GPU Verification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on: {device}")
    
    model = YOLOv4().to(device)
    dummy_input = torch.randn(1, 3, 416, 416).to(device)
    
    outputs = model(dummy_input)
    for i, out in enumerate(outputs):
        # Shape: {N, C, H, W}
        print(f"Stage {i+3} output shape: {out.shape}")