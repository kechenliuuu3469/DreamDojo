from dataclasses import dataclass, field
from typing import (
    Type,
)
import torch
import torch.nn as nn
from torchvision.models.video.s3d import s3d, S3D_Weights
import torchvision.transforms as TF

@dataclass
class VideoS3DEncoderConfig():
    # # target
    # _target: Type = field(default_factory=lambda: VideoS3DEncoder)
    # pre-trained weights to load
    weights = S3D_Weights.KINETICS400_V1
    # option to freeze parameters
    freeze_params: bool = True
    # device
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
# --- Video S3D Encoder ---
class VideoS3DEncoder(nn.Module):
    
    # config: VideoS3DEncoderConfig
    
    def __init__(
        self,
        config: VideoS3DEncoderConfig = VideoS3DEncoderConfig(),
    ):
        super().__init__()
        
        # config
        self.config = config
        
        # device
        self.device = self.config.device
        
        # Initialize S3D using the s3d factory function with pre-trained KINETICS400_V1 weights
        self.backbone = s3d(weights=self.config.weights).to(self.device)

        if self.config.freeze_params:
            # Freeze all parameters of the backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # initialize preprocessing operations
        # expects video input (B, T, C, H, W) or (T, C, H, W)
        # outputs video (..., C, T, H, W)
        # self.preprocessor = S3D_Weights.KINETICS400_V1.transforms
        self.preprocessor = TF.Compose([
            TF.Resize([256, 256], interpolation=TF.InterpolationMode.BILINEAR),
            TF.CenterCrop([224, 224]),
            TF.ConvertImageDtype(torch.float),
            TF.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
            lambda x: x.permute(1, 0, 2, 3) # Permute to (C, T, H, W) if starting from (T, C, H, W)
        ])
    
    # @property
    # def preprocessor(self, img: torch.tensor):
    #     """
    #     Applies the preprocessor transforms 
    #     https://docs.pytorch.org/vision/main/models/generated/torchvision.models.video.s3d.html#:~:text=The%20inference%20transforms,H%2C%20W)%20tensors.
    #     """
    #     preprocessor = TF.Compose([
    #         TF.Resize([256, 256], interpolation=TF.InterpolationMode.BILINEAR),
    #         TF.CenterCrop([224, 224]),
    #         TF.ConvertImageDtype(torch.float),
    #         TF.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
    #         lambda x: x.swapaxes(-3, -4) # Permute to (C, T, H, W) if starting from (T, C, H, W)
    #     ])
        
    def forward(self, x):
        # x should be of shape [B, T, C, H, W]
        # B: Batch size, C: Channels (3 for RGB), T: Frames, H: Height, W: Width
        
        # preprocess video [B, T, C, H, W] to [B, C, T, H, W]
        preprocessed = torch.stack([self.preprocessor(img) for img in x], dim=0)
        
        # compute video features
        return self.backbone(preprocessed)