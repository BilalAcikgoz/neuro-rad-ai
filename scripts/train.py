import torch.nn as nn
import torch
import torch.nn.functional as F

# Memory-optimized 3D UNet model
class LightUNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=2, base_filters=16):
        super(LightUNet3D, self).__init__()
        
        # Use fewer filters to reduce memory usage
        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(inplace=True)
        )
        
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder2 = nn.Sequential(
            nn.Conv3d(base_filters, base_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters*2),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose3d(base_filters*2, base_filters, kernel_size=2, stride=2)
        
        self.decoder1 = nn.Sequential(
            nn.Conv3d(base_filters*2, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        p1 = self.pool1(enc1)
        
        # Bottom level
        enc2 = self.encoder2(p1)
        
        # Decoder
        upconv1 = self.upconv1(enc2)
        
        # Handle size mismatches
        diffY = enc1.size()[2] - upconv1.size()[2]
        diffX = enc1.size()[3] - upconv1.size()[3]
        diffZ = enc1.size()[4] - upconv1.size()[4]
        
        upconv1 = nn.functional.pad(upconv1, [
            diffZ // 2, diffZ - diffZ // 2,
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2
        ])
        
        # Concatenate and decode
        concat1 = torch.cat([upconv1, enc1], dim=1)
        dec1 = self.decoder1(concat1)
        
        # Final classification layer
        out = self.final_conv(dec1)
        
        return out