from __future__ import print_function
import argparse
import torch
from net import FCNet as Net
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from ssp import SSP


def train(args, model, device, dataset, train_kwargs, optimizer, epoch, ssp):
    model.train()
    loss_before = []
    loss_sgd = []
    loss_ssp = []
    train_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss_before_ = loss.item()
        loss_before.append(loss_before_)

        loss.backward()
        optimizer.step()

        loss_sgd.append(F.nll_loss(output, target).item())

        # do SSP
        ssp.step_with_exp_average_gradient(batch_idx, optimizer, device, buffersize=2)


        loss_ssp_ = F.nll_loss(output, target).item()
        loss_ssp.append(loss_ssp_)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_before: {:.6f}\tLoss_sgd: {:.6f}\tLoss_sssp: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss_before_, loss_sgd_, loss_ssp_))
    return loss_before, loss_sgd, loss_ssp


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    # use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_cuda = False

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
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



    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    ssp = SSP()
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    loss ={
        "before":[],
        "sgd":[],
        "ssp":[]
    }
    for epoch in range(1, args.epochs + 1):
        loss_before, loss_sgd, loss_ssp = train(args, model, device, dataset, train_kwargs, optimizer, epoch, ssp)
        loss['before'].extend(loss_before)
        loss["sgd"].extend(loss_sgd)
        loss['ssp'].extend(loss_ssp)
        # scheduler.step()
    with open('loss/ssp+sgd_in_batch_loss','w') as f:
        f.write(str(loss))


if __name__ == '__main__':
    main()