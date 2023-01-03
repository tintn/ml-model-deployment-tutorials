import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
from model import Net


class TraceModel(nn.Module):
    def __init__(self, model, input_size) -> None:
        super().__init__()
        self._model = model
        self.input_size = input_size
    
    def forward(self, imgs):
        # Input is raw images with shape (-1, 3, -1, -1)
        # Preprocess the raw images to match with the expected input size of the model
        imgs = imgs.to(torch.float) / 255.0
        imgs = F.resize(imgs, self.input_size)
        imgs = F.normalize(imgs, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # Feed preprocessed input to the trained model
        logits = self._model(imgs)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return probs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint', '-c',
        help='Path to checkpoint to export', required=True
    )
    parser.add_argument(
        '--output', '-o',
        help='Path to save the exported model', required=True
    )
    return parser.parse_args()


def main():
    args = parse_args()
    net = Net()
    net.load_state_dict(torch.load(args.checkpoint))
    # Create a model with preprocessing included
    model = TraceModel(net, (32, 32))
    
    dummy = torch.randint(0, 256, (1, 3, 32, 32), dtype=torch.uint8)
    traced_model = torch.jit.trace(model, dummy)
    torch.jit.save(traced_model, args.output)

    # Test the traced model
    traced_model = torch.jit.load(args.output)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False,
        download=True)
    correct = 0
    for pil_img, label in testset:
        img = torch.as_tensor(np.array(pil_img, copy=True))
        img = img.view(pil_img.size[1], pil_img.size[0], len(pil_img.getbands()))
        # put it from HWC to CHW format
        img = img.permute((2, 0, 1))
        img = img.unsqueeze(0)
        probs = traced_model(img)[0]
        correct += torch.argmax(probs).item() == label
    print(f'Accuracy of the traced model: {correct / len(testset)}') 

    # Test image with different sizes
    raw_img, _ = testset[0]
    for _ in range(10):
        pil_img = raw_img.copy()
        pil_img = pil_img.resize(np.random.randint(30, 100, 2))
        img = torch.as_tensor(np.array(pil_img, copy=True))
        img = img.view(pil_img.size[1], pil_img.size[0], len(pil_img.getbands()))
        # put it from HWC to CHW format
        img = img.permute((2, 0, 1))
        img = img.unsqueeze(0)
        probs = traced_model(img)[0]
        # The probs should be close
        print(probs)



if __name__ == '__main__':
    main()

