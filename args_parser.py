import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'f-mnist', 'svhn'])
    parser.add_argument('--num_capsules', type=int)
    parser.add_argument('--leaky_routing', action='store_true')
    parser.add_argument('--action', choices=['train', 'eval'])
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--save_step', type=int, default=1000)
    parser.add_argument('--saved_model')
    parser.add_argument('--profile_eval_by_category', action='store_true')
    parser.add_argument('--profile_category', type=int)
    return parser.parse_args()