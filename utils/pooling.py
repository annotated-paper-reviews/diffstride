from typing import Optional, List, Union

import torch
import torch.nn as nn


def compute_adaptive_span_mask(threshold: torch.float,
                               ramp_softness: torch.float,
                               pos: torch.Tensor) -> torch.Tensor:

    output = (1.0 / ramp_softness) * (ramp_softness + threshold - pos)
    return torch.clamp(output, 0.0, 1.0).type(torch.complex64)


class DiffStride(nn.Module):

    def __init__(self,
                strides: List = [2.0, 2.0],
                smoothness_factor: float = 4.0,
                cropping: bool = True,
                lower_limit_stride: Optional[float] = None,
                upper_limit_stride: Optional[float] = None):
        super().__init__()
        self.cropping = cropping
        self.lower_limit_stride = lower_limit_stride
        self.upper_limit_stride = upper_limit_stride
        self.strides = nn.Parameter(torch.tensor(strides))
        self.smoothness_factor = nn.Parameter(torch.tensor(smoothness_factor))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.size()

        horizontal_positions = torch.arange(width // 2 + 1, dtype=torch.float, device=x.device)
        vertical_positions = torch.arange(height // 2 + height % 2, dtype=torch.float, device=x.device)
        vertical_positions = torch.cat(
            [torch.flip(vertical_positions[(height % 2):],dims=(0,)), vertical_positions], 
            axis=0)
        # This clipping by .assign is performed to allow gradient to flow,
        # even when the stride becomes too small, i.e. close to 1.
        min_vertical_stride = height / (height - self.smoothness_factor)
        min_horizontal_stride = width / (width - self.smoothness_factor)
        
        vertical_stride = max(self.strides[0], min_vertical_stride)
        horizontal_stride = max(self.strides[1], min_horizontal_stride)

        strided_height = height / vertical_stride
        strided_width = width / horizontal_stride
        strided_height = max(strided_height, 2.0)
        strided_width = max(strided_width, 2.0)
        lower_height = strided_height / 2.0
        upper_width = strided_width / 2.0 + 1.0

        f_x = torch.fft.rfft2(x)
        horizontal_mask = compute_adaptive_span_mask(
            upper_width, self.smoothness_factor, horizontal_positions)
        vertical_mask = compute_adaptive_span_mask(
            lower_height, self.smoothness_factor, vertical_positions)

        vertical_mask = torch.fft.fftshift(vertical_mask)
        output = f_x * horizontal_mask[None, None, None, :]
        output = output * vertical_mask[None, None, :, None]
        if self.cropping:
            horizontal_to_keep = torch.where(horizontal_mask.type(torch.float) > 0.)[0]
            vertical_to_keep = torch.where(vertical_mask.type(torch.float) > 0.)[0]
            output = output[:, :, vertical_to_keep, :]
            output = output[:, :, :, horizontal_to_keep]
        result = torch.fft.irfft2(output) 

        return result


class DiffStride1d(nn.Module):

    def __init__(self,
                stride: float = 2.0,
                smoothness_factor: float = 4.0,
                cropping: bool = True):
        super().__init__()
        self.cropping = cropping
        self.smoothness_factor = smoothness_factor
        self.stride = nn.Parameter(torch.tensor(stride))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, length = x.size()

        time_positions = torch.arange(length // 2 + 1, dtype=torch.float, device=x.device)
        # This clipping by .assign is performed to allow gradient to flow,
        # even when the stride becomes too small, i.e. close to 1.
        min_time_stride = length / (length - self.smoothness_factor)
        
        time_stride = max(self.stride, min_time_stride)

        strided_time = length / time_stride
        strided_time = max(strided_time, 2.0)
        upper_time = strided_time / 2.0 + 1.0

        f_x = torch.fft.rfft(x)
        time_mask = compute_adaptive_span_mask(
            upper_time, self.smoothness_factor, time_positions)

        output = f_x * time_mask[None, None, :]
        if self.cropping:
            time_to_keep = torch.where(time_mask.type(torch.float) > 0.)[0]
            output = output[:, :, time_to_keep]
        result = torch.fft.irfft(output) 

        return result


if __name__ == "__main__":
    import urllib.request
    import torchvision
    import matplotlib.pyplot as plt

    img_path = "https://upload.wikimedia.org/wikipedia/ko/2/24/Lenna.png"

    with urllib.request.urlopen(img_path) as url:
        with open("lena.png", "wb") as f:
            f.write(url.read())

    img = torchvision.io.read_image('lena.png') / 255.
    img = img.unsqueeze(0)

    pooling = DiffStride()
    x = pooling(img)
    print(img.shape, x.shape)

    fig, axis = plt.subplots(1,2)
    axis[0].imshow(img.squeeze().permute(1,2,0))
    axis[1].imshow(x.squeeze().permute(1,2,0))
    plt.show()
