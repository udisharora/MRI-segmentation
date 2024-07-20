import numpy as np
import torch

class ToTensorMultiChannel(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):
        # Ensure the sample array has the correct shape (H x W x C)
        if sample.ndim == 3:
            # Swap color axis because
            # numpy image: H x W x C
            # torch image: C x H x W
            sample = np.transpose(sample, (2, 0, 1))
            return torch.from_numpy(sample).float()  # Convert to torch tensor and ensure float type
        else:
            raise ValueError(f"Unsupported sample shape: {sample.shape}")

