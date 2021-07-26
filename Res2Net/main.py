import os
import sys
import numpy as np
import argparse
import random
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pdb
from nn_models.ode import FirstOrderODENet, SecondOrderODENet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='ODE')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--input-dim', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--data-size', type=int, default=2000)
    parser.add_argument('--input-noise', type=float, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train-split', type=float, default=0.8)
    parser.add_argument('--total-batches', type=float, default=500)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--model-name', type=str, default='FirstOrderODENet')
    parser.add_argument('--result-path', type=str, default='../results')
    parser.add_argument('--data-path', type=str, default='../data')
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    assert args.input_dim % 2 == 0
    args.device = 'cpu'
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


def draw(stats, save_file_wo_ext):
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams.update({'figure.autolayout': True})
    plt.rc('font', size=23)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
    iterations = np.arange(1, len(stats['train_loss']) + 1)
    axes.plot(iterations, stats['train_loss'][:, 0], label='train')
    axes.plot(iterations, stats['test_loss'][:, 0], label='test')
    axes.set_xlabel('Iterations')
    axes.set_ylabel('Loss')
    ax.grid(True)
    ax.legend(loc='best')
    plt.savefig(save_file_wo_ext + '.png')
    plt.savefig(save_file_wo_ext + '.pdf')
    plt.close()


def train(args):
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)
    data_file_path = os.path.join(args.data_path, 'data.npz')
    if os.path.isfile(data_file_path):
        x = np.load(data_file_path)['x']
        y = np.load(data_file_path)['y']
    else:
        x, y = generate_data(args.input_dim, args.data_size)
        np.savez(data_file_path, x=x, y=y)
    x, y = torch.tensor(x).to(args.device).float(), torch.tensor(y).to(args.device).float()
    x += args.input_noise * torch.randn(*x.shape).to(args.device)
    train_index = np.arange(0, int(x.shape[0] * args.train_split))
    test_index = np.arange(int(x.shape[0] * args.train_split), x.shape[0])
    train_x = x[train_index]
    train_y = y[train_index]
    test_x = x[test_index]
    test_y = y[test_index]

    if args.model_name == 'FirstOrderODENet':
        model = FirstOrderODENet(args.input_dim, args.hidden_dim, args.lr).to(args.device)
    else:
        model = SecondOrderODENet(args.input_dim, args.hidden_dim, args.lr).to(args.device)

    stats = {'train_loss': [], 'test_loss': []}
    with tqdm(total=args.total_batches) as t:
        for step in range(args.total_batches):
            ixs = torch.randperm(train_x.shape[0])[:args.batch_size]
            model.forward_train(train_x[ixs], train_y[ixs])
            train_loss = model.forward_train(train_x, train_y, False, False).cpu().numpy()
            test_loss = model.forward_train(test_x, test_y, False, False).cpu().numpy()
            stats['train_loss'].append([train_loss.mean(), train_loss.std()])
            stats['test_loss'].append([test_loss.mean(), test_loss.std()])
            t.set_postfix(train_loss='{:.9f}'.format(train_loss.mean()), test_loss='{:.9f}'.format(test_loss.mean()))
            t.update()
    stats['train_loss'] = np.array(stats['train_loss'])
    stats['test_loss'] = np.array(stats['test_loss'])
    return stats


def main(args, redirect=True):
    save_path = '../results/' + args.model_name
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if redirect:
        stdout = sys.stdout
        f = open(os.path.join(save_path, 's' + str(args.seed) + '.txt'), 'w+')
        sys.stdout = f
    else:
        f = None
        stdout = None
    stats = train(args)
    if redirect:
        f.close()
        sys.stdout = stdout

    save_stats_file = os.path.join(save_path, 's' + str(args.seed) + '.pl')
    save_drawing_file_wo_ext = os.path.join(save_path, 's' + str(args.seed))
    draw(stats, save_drawing_file_wo_ext)
    with open(save_stats_file, 'w') as f:
        pickle.dump(stats, f, protocol=4)


if __name__ == '__main__':
    main(get_args(), False)
