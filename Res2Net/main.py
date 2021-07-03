import numpy as np
import argparse
import torch
from tqdm import tqdm
import pdb
from nn_models.ode import FirstOrderODENet, SecondOrderODENet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Pendulum-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--input-dim', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--data-size', type=int, default=2000)
    parser.add_argument('--input-noise', type=float, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--total-batches', type=float, default=1000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--model', type=str, default='FirstOrderODENet')
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    assert args.input_dim % 2 == 0
    return args


def generate_data(input_dim, data_size):
    omega = np.random.randn(input_dim // 2).reshape(1, -1)
    phi = np.random.randn(input_dim // 2).reshape(1, -1)
    t = np.arange(data_size).reshape(-1, 1)
    theta_x = omega * t + phi
    theta_y = omega * (t + 1) + phi
    x = np.sin(theta_x)
    dx = omega * np.cos(theta_x)
    y = np.sin(theta_y)
    dy = omega * np.cos(theta_y)
    return np.concatenate((x, dx), axis=1), np.concatenate((y, dy), axis=1)


def main(args):
    x, y = generate_data(args.input_dim, args.data_size)
    x, y = torch.tensor(x).to(args.device).float(), torch.tensor(y).to(args.device).float()
    # model = FirstOrderODENet(args.input_dim, args.hidden_dim, args.lr).to(args.device)
    model = SecondOrderODENet(args.input_dim, args.hidden_dim, args.lr).to(args.device)
    stats = {'train_loss': [], 'test_loss': []}
    with tqdm(total=args.total_batches) as t:
        for step in range(args.total_batches):
            # train step
            ixs = torch.randperm(x.shape[0])[:args.batch_size]
            loss = model.forward_train(x[ixs] + args.input_noise * torch.randn(*x[ixs].shape).to(args.device),
                                       y[ixs] + args.input_noise * torch.randn(*y[ixs].shape).to(args.device))
            train_loss = model.forward_train(x, y, False, False).cpu().numpy()
            # test_loss = model.forward_train(test_x, test_next_x, False, False).cpu().numpy()
            stats['train_loss'].append([train_loss.mean(), train_loss.std()])
            # stats['test_loss'].append([test_loss.mean(), test_loss.std()])
            t.set_postfix(train_loss='{:.9f}'.format(train_loss.mean()))
            # test_loss='{:.9f}'.format(test_loss.mean()))
            # if args.verbose and step % args.print_every == 0:
            #     # run validation
            #     test_ixs = torch.randperm(test_x.shape[0])[:args.batch_size]
            #     test_loss = model.forward_train(test_x[test_ixs], test_next_x[test_ixs], False)
            #     print("step {}, train_loss {:.4e}, test_loss {:.4e}"
            #           .format(step, loss.item(), test_loss.item()))
            t.update()


if __name__ == '__main__':
    main(get_args())
