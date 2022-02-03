"""
    :filename evaluate.py
    :author Tibor Kubik
    :email xkubik34@stud.fit.vutbr.cz

    This script is used to evaluate the network performance or simply to place landmarks on a polygonal model.
    Thus, it can operate in two ways, which can be set by the argument:
        (i)     'Single' mode: [-m|--mode]=single, which loads one STL model into the scene and this model is evaluated,
        (ii)    'Performance' mode: [-m|--mode]=performance, which loads the evaluation dataset part and evaluates
                                    each model one by one. Metrics for the system evaluation are measured and printed
                                    on the standard output.

    Additional arguments needed for the start of the evaluation (excluding the already discussed mode argument):
        (i)     [-p|--path]=path:           path to the STL file in single mode, or to the folder with evaluation
                                            dataset in the performance mode.
        (ii)    [-n|--network-path]=path:   path to the trained network which will be used for the evaluation.
        (iii)   [-nt|--network-name]=name:  name of the network. This must be one of BatchNormUNet,
                                            AttentionUNet, or NestedUNet.

    Controls:
        Mouse   -   panning, zooming, shifting in the scene, so the model
                    is in requisite position before evaluation starts.
        G       -   starts the evaluation of the polygon mesh - landmarks are placed on its surface.
        Tab     -   load next polygonal model for evaluation.
"""

import glob
import torch
import argparse

from src import config
from networks.UNet import UNet
from networks.NestedUNet import NestedUNet
from networks.AttUNet import AttUNet
from evaluator.Evaluator import Evaluator
from evaluator.PerformanceMeasure import PerformanceMeasure


if __name__ == '__main__':

    '''Argument parsing.'''
    parser = argparse.ArgumentParser(description='Evaluation of teeth landmark detection',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--mode', metavar='M', type=str, default='performance',
                        help='Mode of evaluation. Either single polygon, or performance evaluation', dest='mode')
    parser.add_argument('-p', '--path', metavar='P', type=str, default=None,
                        help='Path to polygon if single mode, to the folder with STLs if performance', dest='path')
    parser.add_argument('-n', '--network-path', metavar='N', type=str, default=None,
                        help='Path to the network', dest='network_path')
    parser.add_argument('-nt', '--network-name', metavar='NT', type=str, default=None,
                        help='Network name', dest='network_name')
    parser.add_argument('-i', '--input-format', metavar='I', type=str, default='depth',
                        help='Network input format (what will be rendered)', dest='input_format')
    args = parser.parse_args()

    assert args.mode in ['single', 'performance']
    assert args.network_name in ['BatchNormUNet', 'AttentionUNet', 'NestedUNet']
    assert args.input_format in ['depth', 'geom', 'depth+geom']

    in_channels = 1

    if args.input_format == 'depth+geom':
        in_channels = 2

    with torch.no_grad():
        '''Chosen network initializing + state loading.'''
        if args.network_name == 'BatchNormUNet':
            network = UNet(in_channels=in_channels, out_channels=config.LANDMARKS_NUM, batch_norm=True, decoder_mode='upconv')
        elif args.network_name == 'AttentionUNet':
            network = AttUNet(in_channels=in_channels, out_channels=config.LANDMARKS_NUM)
        else:
            network = NestedUNet(in_channels=in_channels, out_channels=config.LANDMARKS_NUM)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        network.to(device)

        network.load_state_dict(torch.load(args.network_path))
        network.eval()

        '''Evaluation start according to the evaluation mode.'''
        if args.mode == 'single':
            '''No need for PerformanceMeasure instance in single mode.'''
            evaluator = Evaluator(network=network, model_path=args.path, single=True, input_format=args.input_format)
        else:
            model_paths = glob.glob(args.path + '*.stl')
            perf_meas = PerformanceMeasure(len(model_paths))

            evaluator = Evaluator(network=network, model_path=model_paths, perf_meas=perf_meas, input_format=args.input_format)

        '''Keep this line here to keep the scene window open.'''
        evaluator.interactor.Start()
