"""
    :filename train.py
    :author Tibor Kubik
    :email xkubik34@stud.fit.vutbr.cz

    This script is used to train a neural network on the dataset of 2D depth maps of dentition.
    Goal is to train a neural network in a regression way, which can predict landmarks in a form of heatmaps.

    The training hyperparameters as well as the neural network architecture selection can be set in two ways:
        (i) By modifying the config.py file, which contains dictionaries with the hyperparameter values,
        (ii) or by provided script arguments.

        The only obligatory argument is the neural network architecture, which can be one of:
        (i) Batch Normalization U-Net: argument [-n|--network-name]=BatchNormUNet,
        (ii) Attention U-Net: argument [-n|--network-name]=AttUNet,
        (iii) or Nested U-Net: argument [-n|--network-name]=NestedUNet.
"""

import torch
import logging
import argparse

from torchsummary import summary

import config
from networks.UNet import UNet
from trainer.Trainer import Trainer
from networks.AttUNet import AttUNet
from networks.NestedUNet import NestedUNet


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    '''Argument parsing for the config overwrite.'''
    parser = argparse.ArgumentParser(description='UNet for teeth landmark detection training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int,
                        default=config.HYPERPARAMETERS['epochs'], help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?',
                        default=config.HYPERPARAMETERS['batch_size'], help='Batch size', dest='batch_size')
    parser.add_argument('-w', '--weight-decay', metavar='WD', type=float, nargs='?',
                        default=config.HYPERPARAMETERS['weight_decay'], help='Weight decay', dest='weight_decay')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?',
                        default=config.HYPERPARAMETERS['learning_rate'], help='Learning rate', dest='lr')
    parser.add_argument('-n', '--network-name', metavar='NN', type=str,
                        default='BatchNormUNet', help='Network name', dest='network_name')
    parser.add_argument('-i', '--input-format', metavar='IF', type=str,
                        default='depth', help='Input type (depth, geom, depth+geom', dest='input_format')
    parser.add_argument('-f', '--folder-path', metavar='FP', type=str, help='Folder path to prepared'
                                                                            '2D renders and landmark heatmaps',
                        dest='folder_path')
    args = parser.parse_args()

    assert args.network_name in ['BatchNormUNet', 'AttUNet', 'NestedUNet']
    assert args.input_format in ['depth', 'geom', 'depth+geom']

    '''Better GPU performance.'''
    torch.backends.cudnn.benchmark = True

    in_channels = 1

    if args.input_format == 'depth+geom':
        in_channels = 2

    '''Creation of chosen network instance.'''
    if args.network_name == 'BatchNormUNet':
        network = UNet(in_channels=in_channels, out_channels=config.LANDMARKS_NUM, batch_norm=True, decoder_mode='upconv')
    elif args.network_name == 'AttUNet':
        network = AttUNet(in_channels=in_channels, out_channels=config.LANDMARKS_NUM)
    else:
        network = NestedUNet(in_channels=in_channels, out_channels=config.LANDMARKS_NUM)

    '''Device setup - strongly recommend the usage of NVIDIA GPU (cuda).'''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    network.to(device)

    '''
    Device and network information.
    Please uncomment last line if you want to check the network layers in the log window.
    '''
    logging.info(f'Device used: {device}')
    logging.info(f'GPU name: {torch.cuda.get_device_name(0)}')
    # logging.info(summary(network, input_size=(1, config.DIMENSIONS['input_net'], config.DIMENSIONS['input_net'])))

    try:
        trainer = Trainer(network=network,
                          device=device,
                          epochs=args.epochs,
                          batch_size=args.batch_size,
                          weight_decay=args.weight_decay,
                          lr=args.lr,
                          input_type=args.input_format,
                          dataset_path=args.folder_path)

        trainer.training()

    except KeyboardInterrupt:
        torch.save(network.state_dict(), 'interrupted_model.pt')
