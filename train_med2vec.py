#################################################################
# Code written by Sajad Darabi (sajad.darabi@cs.ucla.edu)
# For bug report, please contact author using the email address
############################################### ##################
import importlib
import os
import json
import argparse
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
from model.med2vec import Med2Vec
import torch.multiprocessing as mp
from trainer import Med2VecTrainer as Trainer
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def get_instance(module, name, config, *args):
    if name == 'Med2Vec':
        return module(*args, **config['model'])
    else:
        return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume, device):

    # setup data_loader instances
    data_loader = get_instance(module_data, 'data_loader', config)

    # build model architecture
    model_args = config['model']['args']
    # TODO: load the device with the parser and not manualy like here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Med2Vec(**model_args,device=device)

    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
    lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)

    

    trainer = Trainer(model, loss, metrics, optimizer,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      lr_scheduler=lr_scheduler,
                      train_logger=None)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default="configs/config.json", type=str,
                           help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=int,
                           help='index of the GPU to use (default: None)')
    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    if args.device is not None:
        # Set the device to the specified GPU
        device = torch.device(f"cuda:{args.device}")
    else:
        # Use CPU if no device is specified
        device = torch.device("cpu")
        
    mp.set_start_method('spawn')


    main(config, args.resume, device)