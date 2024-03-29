from collections import defaultdict

class ActivationStorage:
    def __init__(self):
        self.data = defaultdict(list)

    def hook(self, module, input, output):
        if type(output) is tuple:
            self.data[module].append(output[0].detach().cpu())
        else:
            self.data[module].append(output.detach().cpu())

    def register_activation_hooks(self, model):
        for name, module in model.named_modules():
            if len([m for m in module.modules()]) == 1: # considering only the lowest-level modules
                module.register_forward_hook(self.hook)
            

class GradStorage:
    def __init__(self):
        self.data = defaultdict(list)
        
    def gather_grads(self, model):
        for name, param in model.named_parameters():
            self.data[name].append(param.grad)


if __name__ == "__main__":
    import torch
    import torch.nn as nn

    # model definition & instantiation
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
        def forward(self, x):
            return self.layers(x)
    
    model = Model()

    # attaching hooks for activation
    activation_storage = ActivationStorage()
    activation_storage.register_activation_hooks(model)

    # executing forward pass
    for _ in range(3):
        dummy_input = torch.randn((2, 1024))
        y = model(dummy_input)
        
    # executing backward pass
    y.sum().backward()

    # gather grads
    grad_storage = GradStorage()
    grad_storage.gather_grads(model)

    # checking stored activation
    for k, v in activation_storage.data.items():
        print(k)
        print(f'\t{len(v)} activations stored with the shape of {v[0].shape}')

    # checking stored gradients
    for k, v in grad_storage.data.items():
        print(k)
        print(f'\t{len(v)} grads stored with the shape of {v[0].shape}')