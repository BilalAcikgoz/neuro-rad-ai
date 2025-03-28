import torch.nn as nn
import torch
import torch.nn.functional as F

# Improved 3D UNet with residual connections and deeper architecture
class ImprovedUNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=2, base_filters=16):
        super(ImprovedUNet3D, self).__init__()
        
        # Encoder path with residual blocks
        self.enc1 = self._make_layer(in_channels, base_filters)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = self._make_layer(base_filters, base_filters*2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc3 = self._make_layer(base_filters*2, base_filters*4)
        
        # Decoder path with skip connections
        self.upconv2 = nn.ConvTranspose3d(base_filters*4, base_filters*2, kernel_size=2, stride=2)
        self.dec2 = self._make_layer(base_filters*4, base_filters*2)
        
        self.upconv1 = nn.ConvTranspose3d(base_filters*2, base_filters, kernel_size=2, stride=2)
        self.dec1 = self._make_layer(base_filters*2, base_filters)
        
        # Output layer with dropout for regularization
        self.dropout = nn.Dropout3d(0.3)
        self.final_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)
        
        # Initialize weights for better convergence
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels):
        """Create a residual block"""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),  # More stable than BatchNorm for small batches
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None:  # Eğer weight tanımlıysa
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:  # Eğer bias tanımlıysa
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)
        p1 = self.pool1(enc1_out)
        
        enc2_out = self.enc2(p1)
        p2 = self.pool2(enc2_out)
        
        # Bottom level
        enc3_out = self.enc3(p2)
        
        # Decoder with skip connections
        up2 = self.upconv2(enc3_out)
        # Ensure sizes match for skip connection
        diffY = enc2_out.size()[2] - up2.size()[2]
        diffX = enc2_out.size()[3] - up2.size()[3]
        diffZ = enc2_out.size()[4] - up2.size()[4]
        
        up2 = F.pad(up2, [
            diffZ // 2, diffZ - diffZ // 2,
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2
        ])
        concat2 = torch.cat([up2, enc2_out], dim=1)
        dec2_out = self.dec2(concat2)
        
        up1 = self.upconv1(dec2_out)
        # Ensure sizes match for skip connection
        diffY = enc1_out.size()[2] - up1.size()[2]
        diffX = enc1_out.size()[3] - up1.size()[3]
        diffZ = enc1_out.size()[4] - up1.size()[4]
        
        up1 = F.pad(up1, [
            diffZ // 2, diffZ - diffZ // 2,
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2
        ])
        concat1 = torch.cat([up1, enc1_out], dim=1)
        dec1_out = self.dec1(concat1)
        
        # Apply dropout for regularization
        x = self.dropout(dec1_out)
        
        # Final classification layer
        out = self.final_conv(x)
        
        return out