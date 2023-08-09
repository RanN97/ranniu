import argparse

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size_transformer', type=int, default=8)
    parser.add_argument('--decoder_num_layers', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--transformer_dropout', type=float, default=0.1)
    parser.add_argument('--transformer_nhead', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--adam_weight_decay', type=float, default=0.01)
    parser.add_argument('--epoch', type=int, default=100)

    # args = parser.parse_args(args=['--device', '0', '--no_cuda'])
    args, unknown = parser.parse_known_args()
    return args