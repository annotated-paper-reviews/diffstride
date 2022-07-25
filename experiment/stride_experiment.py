import torch
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt

model = models.resnet18(pretrained=True)

downsamplers = [
    model.conv1,
    model.layer2[0].conv1,
    model.layer2[0].downsample[0], # maxpooling
    model.layer3[0].conv1,
    model.layer3[0].downsample[0], # maxpooling
    model.layer4[0].conv1,
    model.layer4[0].downsample[0], # maxpooling
]

fig1, axs1 = plt.subplots(len(downsamplers), 3, constrained_layout=True)
fig1.set_size_inches(5, 8.5)
fig2, axs2 = plt.subplots(len(downsamplers), 3, constrained_layout=True)
fig2.set_size_inches(5, 8.5)
for i, downsampler in enumerate(downsamplers):
    # prepare params
    stride = downsampler.stride
    padding = downsampler.padding
    o_c, i_c, _, _ = downsampler.weight.shape 
    whitenoise = torch.randn(1, i_c, 64, 64)
    print('input shape:', whitenoise.shape)
    
    # generate output
    conv_kernel = downsampler.weight
    conv_output = F.conv2d(whitenoise, conv_kernel, bias=None, 
                           stride=stride, padding=padding)
    ones_kernel = torch.ones_like(conv_kernel)
    ones_output = F.conv2d(whitenoise, ones_kernel, bias=None, 
                           stride=stride, padding=padding)    
    rand_kernel = torch.randn(conv_kernel.shape)
    rand_output = F.conv2d(whitenoise, rand_kernel, bias=None, 
                           stride=stride, padding=padding)
    print('output shape:', conv_output.shape, '\n')

    # generate fft
    conv_fft = torch.abs(torch.fft.fftshift(torch.fft.fft2(conv_output[0,0,:,:])))
    ones_fft = torch.abs(torch.fft.fftshift(torch.fft.fft2(ones_output[0,0,:,:])))
    rand_fft = torch.abs(torch.fft.fftshift(torch.fft.fft2(rand_output[0,0,:,:])))

    # plot
    axs1[i, 0].imshow(conv_fft.detach().numpy())
    axs1[i, 1].imshow(ones_fft.detach().numpy())
    axs1[i, 2].imshow(rand_fft.detach().numpy())
    axs1[i, 0].title.set_text('conv_fft ' + str(conv_kernel.shape[-2]) + str(conv_kernel.shape[-2]))
    axs1[i, 1].title.set_text('ones_fft ' + str(conv_kernel.shape[-2]) + str(conv_kernel.shape[-2]))
    axs1[i, 2].title.set_text('rand_fft ' + str(conv_kernel.shape[-2]) + str(conv_kernel.shape[-2]))

    axs2[i, 0].plot(conv_fft.detach().numpy()[16,:])
    axs2[i, 1].plot(ones_fft.detach().numpy()[16,:])
    axs2[i, 2].plot(rand_fft.detach().numpy()[16,:])
    axs2[i, 0].title.set_text('conv_fft ' + str(conv_kernel.shape[-2]) + str(conv_kernel.shape[-2]))
    axs2[i, 1].title.set_text('ones_fft ' + str(conv_kernel.shape[-2]) + str(conv_kernel.shape[-2]))
    axs2[i, 2].title.set_text('rand_fft ' + str(conv_kernel.shape[-2]) + str(conv_kernel.shape[-2]))

plt.show()
plt.close()

