# from __future__ import print_function
import argparse
import torch
from net import FCNet as Net
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# torch.set_default_dtype(torch.float64)
# from utils import compute_denom, compute_numer, update_parameter, getWeightAndGrad, check_value


def getGrad(optimizer):
    grads = []

    for group in optimizer.param_groups:
        grads_ = []
        for p in group['params']:
            if p.grad is not None:
                grads_.append(p.grad.clone())
        grads.append(grads_)

    return grads



def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        true_gradient = getGrad(optimizer)
        itl = int(train_loader.batch_size / 5)
        a =0
        for i in range(5):
            optimizer.zero_grad()
            data_ , target_ = data[itl*i:itl*i+itl],  target[itl*i:itl*i+itl]
            output = model(data_)
            loss_ = F.nll_loss(output, target_)
            loss_.backward()

            g = getGrad(optimizer)
            a += g[0][0][0][0].item()
            print(loss_.item(), a)

        print("#######################")
        optimizer.zero_grad()
        for i in range(5):

            data_ , target_ = data[itl*i:itl*i+itl],  target[itl*i:itl*i+itl]
            # print(len(data_))
            output = model(data_)
            loss_ = F.nll_loss(output, target_)

            loss_.backward()
            g = getGrad(optimizer)
            print(loss_.item(), g[0][0][0][0].item())


        gradient = getGrad(optimizer)
        print((gradient[0][0][0][0]/5).item())
        print(true_gradient[0][0][0][0].item())
        return None






def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=1000, metavar='N')
    parser.add_argument('--epochs', type=int, default=1, metavar='N')
    parser.add_argument('--seed', type=int, default=1, metavar='S')

    use_cuda = torch.cuda.is_available()
    args = parser.parse_args()
    # use_cuda = False

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)


    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('../data', train=True,
                              transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset,**train_kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer)



if __name__ == '__main__':
    main()