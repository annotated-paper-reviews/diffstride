import os
import argparse
from argparse_dataclass import ArgumentParser
from dataclasses import dataclass, field
import logging

import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

import torch.optim as optim
from models.m5 import M5
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

"""
https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html
"""

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

logger = logging.getLogger(__name__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("*"*50)
print(f'Detected device: {device}')
print("*"*50)

@dataclass
class Arguments:
    """
    Arguments pertaining to training
    """
    batch_size: int = field(default=256,)
    num_epochs: int = field(default=10,)
    log_step: int = field(default=50,)
    log_path: str = field(default='./runs')
    data_path: str = field(default='./data',)
    learning_rate: float = field(default=0.01,)
    momentum: float = field(default=0.9,)


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None, data_path: str = "./data"):
        super().__init__(data_path, download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

def main():
    # args
    parser = ArgumentParser(Arguments)
    args = parser.parse_args([])
    
    # logger
    writer = SummaryWriter(args.log_path)

    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False
    
    train_set = SubsetSC(subset="training", data_path=args.data_path)
    valid_set = SubsetSC(subset="validation", data_path=args.data_path)
    test_set = SubsetSC(subset="testing", data_path=args.data_path)
    
    waveform, sample_rate, utterance, *_ = train_set[-1]
    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))

    def label_to_index(word):
        # Return the position of the word in labels
        return torch.tensor(labels.index(word))

    def index_to_label(index):
        # Return the word corresponding to the index in labels
        # This is the inverse of label_to_index
        return labels[index]

    def pad_sequence(batch):
        # Make all tensor in a batch the same length by padding with zeros
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)

    def collate_fn(batch):
        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number
        tensors, targets = [], []

        # Gather in lists, and encode labels as indices
        for waveform, _, label, *_ in batch:
            tensors += [waveform]
            targets += [label_to_index(label)]

        # Group the list of tensors into a batched tensor
        tensors = pad_sequence(tensors)
        targets = torch.stack(targets)

        return tensors, targets

    def number_of_correct(pred, target):
        # count number of correct predictions
        return pred.squeeze().eq(target).sum().item()

    def get_likely_index(tensor):
        # find most likely label index for each element in the batch
        return tensor.argmax(dim=-1)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=8000)
    transformed = transform(waveform)
    transform.to(device)

    print("*"*50)
    print('Model initialized')
    print("*"*50)
    model = M5(n_input=transformed.shape[0], n_output=len(labels), downsample_type="diffstride")
    model.to(device)
    #riter.add_graph(model, transformed.to(device))

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    print("*"*50)
    print(f'Training...')
    print("*"*50)
    step = 0
    for epoch in range(1, args.num_epochs + 1):
        # train
        model.train()
        for data, target in train_loader:
            
            data = data.to(device)
            target = target.to(device)

            # apply transform and model on whole batch directly on device
            data = transform(data)
            output = model(data)

            # negative log-likelihood for a tensor of size (batch x 1 x n_output)
            loss = F.nll_loss(output.squeeze(), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print training stats
            if step % args.log_step == 0:
                print(f"Train Epoch: {epoch}, step: {step} \tLoss: {loss.item():.6f}")
                writer.add_scalar(
                    'training loss',
                    loss.item(),
                    step
                )
            step += 1
        
        # validate
        model.eval()
        correct = 0
        for data, target in valid_loader:

            data = data.to(device)
            target = target.to(device)

            # apply transform and model on whole batch directly on device
            data = transform(data)
            output = model(data)

            pred = get_likely_index(output)
            correct += number_of_correct(pred, target)
        print(f"\nValidation Epoch: {epoch}\tAccuracy: {correct}/{len(valid_loader.dataset)} ({100. * correct / len(valid_loader.dataset):.0f}%)\n") 
        scheduler.step()

    # test
    model.eval()
    correct = 0
    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)
    print(f"\nTest Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")


if __name__ == '__main__':
    main()





