import argparse
import torch
from net import FCNet as Net
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time

torch.set_default_dtype(torch.float64)
# float32
# average over loss[multi backward] 	 0.000767876161262393 6.303839206695557
# average over loss[one backward] 	 0.0007678759866394103 6.2055909633636475

#float64
# average over loss[multi backward] 	 1.9961114481237086e-06 6.716878890991211
# average over loss[one backward] 	 1.9961114481236662e-06 7.701658248901367

# float16
# average over loss[multi backward] 	 8.845329284667969e-05 6.699422836303711
# average over loss[one backward] 	 8.857250213623047e-05 6.65080714225769

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
    begin = time.time()
    optimizer.zero_grad()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

    gradient = getGrad(optimizer)
    print("average over loss[multi backward] \t", gradient[0][0][0][0].item(), time.time()-begin)

    a = 0
    begin= time.time()
    optimizer.zero_grad()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = F.nll_loss(output, target)
        a+=loss
    a.backward()

    gradient = getGrad(optimizer)
    print("average over loss[one backward] \t", gradient[0][0][0][0].item(), time.time()-begin)







def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=10000, metavar='N')
    parser.add_argument('--epochs', type=int, default=1, metavar='N')
    parser.add_argument('--seed', type=int, default=1, metavar='S')

    use_cuda = torch.cuda.is_available()
    args = parser.parse_args()
    # use_cuda = False

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
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