#!/usr/bin/env python

import torch
import argparse

class AsyncNone(object):
    def __await__(self):
        yield


def get_data_folder(args):
    
    folder = args.train_name + "_"
    folder += args.model_arch + "_" 
    folder += ("reverse" if args.mcts_reverse_q else "normal") + "_"
    folder += ("normalize" if args.game_state_normalize else "regular") + "_"
    folder += str(args.model_channels) + "_"
    folder += str(args.game_size)
    return folder


def get_model_filepath(args):

    return get_data_folder(args) + '/model.data'


def get_best_model_filepath(args):

    return get_data_folder(args) + '/model.best'


def parse_args():
    
    parser = argparse.ArgumentParser(description='mcts')
    parser.add_argument('--game_size',           type=int,   default=11,          help='game size (default: 11)')
    parser.add_argument('--game_n_in_a_row',     type=int,   default=5,           help='n_in_a_row (default: 5)')
    parser.add_argument('--game_state_normalize',type=int,   default=0,           help='normalize state (default: 0)')
    parser.add_argument('--model_batch_size',    type=int,   default=4096,        help='batch size (default: 4096)')
    parser.add_argument('--model_arch',          type=str,   default='conv3',     help='model arch (conv3, conv4, resnet)')
    parser.add_argument('--model_ppo2',          type=int,   default=0,           help='model using ppo2 (default 0)')
    parser.add_argument('--model_ppo2_clip',     type=float, default=0.2,         help='model ppo2 clip value (default 0.2)')
    parser.add_argument('--model_coef_value',    type=float, default=1.0,         help='coeffiency for value loss (default 1.0)')
    parser.add_argument('--model_coef_entropy',  type=float, default=0.05,        help='coeffiency for entropy (default 0.05)')
    parser.add_argument('--model_tensorrt',      type=int,   default=0,           help='model use tensorrt (default 0)')
    parser.add_argument('--model_precision',     type=str,   default='float',     help='model precision (default float)')
    parser.add_argument('--model_channels',      type=int,   default=32,          help='resnet channels (default: 32)')
    parser.add_argument('--model_cuda',          type=int,   default=-1,          help='whether to use cuda (default: -1)')
    parser.add_argument('--model_epochs',        type=int,   default=8,           help='number of epoch to train (default: 8)')
    parser.add_argument('--model_lr_base',       type=float, default=0.002,       help='learning rate (default: 0.002)')
    parser.add_argument('--model_lr_range',      type=float, default=50.0,        help='learning rate range (default: 50.0)')
    parser.add_argument('--model_kl_target',     type=float, default=0.02,        help='learning rate (default: 0.02)')
    parser.add_argument('--model_weight_decay',  type=float, default=1e-4,        help='weight decay (default: 1e-4)')
    parser.add_argument('--mcts_playout',        type=int,   default=800,         help='mcts playout param (default: 800)')
    parser.add_argument('--mcts_cpuct',          type=float, default=10.0,        help='mcts bound param (default: 10.0)')
    parser.add_argument('--mcts_gamma',          type=float, default=0.98,        help='mcts gamma param (default: 0.98)')
    parser.add_argument('--mcts_alpha',          type=float, default=0.2,         help='mcts alpha param (default: 0.2)')
    parser.add_argument('--mcts_noise',          type=float, default=0.25,        help='mcts noise param (default: 0.25)')
    parser.add_argument('--mcts_temperature',    type=float, default=1.0,         help='mcts temperature (default: 1.0)')
    parser.add_argument('--mcts_debug',          type=int,   default=0,           help='mcts debug level (default: 0)')
    parser.add_argument('--mcts_reverse_q',      type=int,   default=1,           help='mcts reverse q (default: 1)')
    parser.add_argument('--mcts_sleep_int',      type=int,   default=5,           help='mcts sleep interval (default: 5)')
    parser.add_argument('--mcts_display',        type=bool,  default=False,       help='mcts display (default: false)')
    parser.add_argument('--train_name',          type=str,   default="save",      help='training data size (default: "save")')
    parser.add_argument('--train_game_count',    type=int,   default=3,           help='training game count (default: 3)')
    parser.add_argument('--train_save_count',    type=int,   default=20,          help='training save count (default: 20)')
    parser.add_argument('--train_server',        type=bool,  default=False,       help='training server mode (default: False)')
    parser.add_argument('--memory_min',          type=int,   default=20000,       help='training memory min (default: 20000)')
    parser.add_argument('--memory_max',          type=int,   default=400000,      help='training memory max (default: 400000)')
    parser.add_argument('--memory_server',       type=str,   default="localhost", help='training server (default: "localhost")')
    parser.add_argument('--memory_port',         type=str,   default="8080",      help='training port (default: "8080")')
    parser.add_argument('--play_dtype',          type=str,   default="float32",   help='play dtype (default: float32)')
    parser.add_argument('--play_model_refresh',  type=int,   default=10,          help='play model refresh (default: 10)')
    parser.add_argument('--play_post_exp',       type=int,   default=1,           help='play post experience (default: 1)')
    parser.add_argument('--eval_game_count',     type=int,   default=18,          help='evaluate game count (default: 18)')
    parser.add_argument('--eval_model_1',        type=str,   default="curr",      help='eval model 1 (best, curr, human, online)')
    parser.add_argument('--eval_model_2',        type=str,   default="best",      help='eval model 2 (best, curr, human, online)')
    parser.add_argument('--eval_best_threshold', type=float, default=0.67,        help='best model threshold (default: 0.67)')
    parser.add_argument('--timeit_count',        type=int,   default=100,         help='training data size (default: 100)')

    args = parser.parse_args()

    args.model_cuda = args.model_cuda if torch.cuda.is_available() else -1
    
    return args
    


if __name__ == "__main__":
    args = parse_args()
    print(args)

