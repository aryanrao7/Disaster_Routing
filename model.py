import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    """The core engine block: (Convolution => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Build the Encoder (Left side of the U)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Build the Decoder (Right side of the U)
        for feature in reversed(features):
            # ConvTranspose2d scales the image resolution back up
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        # The Bottleneck (Bottom of the U)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # Final output layer (Collapses features down to 1 channel for the binary mask)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # --- DOWNWARD PATH (Encoder) ---
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # --- THE BOTTLENECK ---
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # Reverse the saved features list

        # --- UPWARD PATH (Decoder) ---
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x) # Scale the image up
            skip_connection = skip_connections[i//2] # Grab the saved features from the Encoder

            # Failsafe: If image dimensions got weirdly rounded, resize the skip connection to match
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            # SKIP CONNECTION: Stitch the high-res encoder features onto the upscaled decoder features
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i+1](concat_skip) # Run through double convolution

        # --- OUTPUT ---
        # We output raw mathematical logits. (We will apply Sigmoid later to turn them into 0-1 probabilities)
        return self.final_conv(x)

# --- Quick Test ---
if __name__ == "__main__":
    print("Testing U-Net Architecture...")
    # Create a dummy image tensor (Batch Size=1, Channels=3 (RGB), Height=256, Width=256)
    x = torch.randn((1, 3, 256, 256))
    model = UNet(in_channels=3, out_channels=1)
    
    # Run the dummy image through the model
    predictions = model(x)
    
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {predictions.shape}")
    if predictions.shape == (1, 1, 256, 256):
        print("SUCCESS! The architecture works. It ingested an RGB image and spit out a 1-channel mask.")
