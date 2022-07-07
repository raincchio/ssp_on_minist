from __future__ import print_function
import argparse
import torch
from net import FCNet as Net
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
# from utils import compute_denom, compute_numer, update_parameter, getWeightAndGrad, check_value


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    return losses





def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=128, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N')
    parser.add_argument('--epochs', type=int, default=200, metavar='N')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M')
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N')

    use_cuda = torch.cuda.is_available()
    args = parser.parse_args()
    # use_cuda = False

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('data', train=True,
                              transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset,**train_kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters())
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    f = open('loss/adam_loss','w')
    for epoch in range(1, args.epochs + 1):
        loss_ = train(args, model, device, train_loader, optimizer, epoch)
        f.write(str(loss_)+'\n')
        # scheduler.step()
    f.close()


if __name__ == '__main__':
    main()