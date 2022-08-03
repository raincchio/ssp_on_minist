import argparse
import os.path

import torch
from net import FCNet as Net

import torch.optim as optim
from torchvision import datasets, transforms
from ssp import SSP
from utils import train_EP, test, train_GD

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--beta1', type=float, default=0.9, metavar='B1',
                        help='adam beta1 value (default: 1.0)')
    parser.add_argument('--beta2', type=float, default=0.999, metavar='B2',
                        help='adam beta2 value (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--ssp', action='store_true', default=False, help='step size planning')
    parser.add_argument('--path', type=str, default="loss/", metavar='F', help='realtive loss file path')
    parser.add_argument('--optimizer', type=str, default="sgd", metavar='O', help='optimizer')
    parser.add_argument('--noise', action='store_true', default=False,  help='noise.')
    parser.add_argument('--bufferlength', type=int, default=3, metavar='K', help='bufferlength')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    print(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # use_cuda = False

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': 1000}
    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                       'pin_memory': False,
                       'shuffle': False}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset_train = datasets.MNIST('data', train=True, transform=transform)
    dataset_test = datasets.MNIST('data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)


    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # lr 1e-3 1e-4
    # b1 0.9 .99
    # b2 0.999 0.9999
    # adam default lr=1e-3, betas=(0.9, 0.999)

    model = Net().to(device)
    assert args.optimizer in ['adam', 'sgd'], ' unsupport optimizer'
    if args.optimizer=='adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        fname = args.optimizer + "-lr_" + str(args.lr) + "-b1_" + str(args.beta1) + "-b2_" + str(args.beta2)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        fname = args.optimizer + "-lr_" + str(args.lr)

    # optimizer = optim.RMSprop()

    ssp = None
    if args.ssp:
        ssp = SSP()
        fname = 'ssp-'+fname + '-K_'+str(args.bufferlength)

        if args.noise:
            fname += "-noise"

    fpath = "loss/0803/"
    os.makedirs(fpath+'alpha', exist_ok=True)
    print(fpath+fname)
    f = open(fpath+fname, "w")
    # fc = open("loss/ssp+sgd_in_epoch_loss_compare","w")

    for epoch in range(1, args.epochs+1):
        # use any grdaient descent method for optimizer
        train_loss = train_EP(args, model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)

        # step size planning  between consequcence 3 epoch parameter
        extra = 0
        if args.ssp:
            extra = ssp.step_with_true_gradient(model, device, dataset_train, optimizer,epoch, fpath, K=args.bufferlength, noise=args.noise)


        f.write(str([train_loss, test_loss, extra])+'\n')

        # f.write(str(train_loss) + '\n')

if __name__ == '__main__':
    main()
    # python run_ssp_in_epoch.py --lr=1e-2 --beta1=0.9 --beta2=0.999 --ssp
    # python run_ssp_in_epoch.py --optimizer=sgd --ssp --lr=1e-2 --noise
    # python run_ssp_in_epoch.py --optimizer=sgd --ssp --lr=1e-2 --bufferlength 5

    # cchio minist
    # cd workspace/minist
    # python run_ssp_in_epoch.py --ssp --optimizer=sgd --lr=0.1 --bufferlength=1