"""
    :filename Trainer.py
    :author Tibor Kubik
    :email xkubik34@stud.fit.vutbr.cz

    Class Trainer.

    This class is the core of the training procedure of proposed method.
    The training and validation loop is located in this source file.
"""

import time
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

from utils.utils_2d import *
from src.trainer.RMSELoss import RMSELoss
from src.trainer.transformations import *
from src.trainer.CBRMSELoss import CBRMSELoss
from src.trainer.JawDataset import JawDataset
from src.trainer.EarlyStopping import EarlyStopping
from src.trainer.MissingTeethSampler import MissingTeethSampler


class Trainer:
    """ Trainer class - the training and validation of the network."""

    def __init__(self, network, device, epochs, batch_size, weight_decay, lr, input_type, dataset_path):
        self.network = network
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.lr = lr
        self.input_type = input_type  # NN input type => depth/geometry/combination...
        self.dataset_path = dataset_path

        self.criterion = CBRMSELoss()

        self.difficulties = {
            '1-and-8-present': [],
            '1-present': [],
            '8-present': [],
            'missing-both': []
        }  # difficulties in learning

        self.weights = {
            '1-and-8-present': 1.0,
            '1-present': 1.0,
            '8-present': 1.0,
            'missing-both': 1.0
        }

        self.optimizer = optim.Adam(self.network.parameters(), weight_decay=self.weight_decay, lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              verbose=True,
                                                              patience=config.HYPERPARAMETERS['lr_scheduler_patience'],
                                                              min_lr=config.HYPERPARAMETERS['lr_scheduler_min_lr'],
                                                              factor=config.HYPERPARAMETERS['lr_scheduler_factor'])
        self.scaler = torch.cuda.amp.GradScaler()
        self.early_stopping = EarlyStopping(patience=config.HYPERPARAMETERS['early_stopping_patience'], verbose=True)
        self.stop_flag = False

        self.running_train_loss = 0.0
        self.running_val_loss = 0.0
        self.training_loss_list = []
        self.validation_loss_list = []

        self.dataset = JawDataset(dataset_path=self.dataset_path,
                                  transform=transforms.Compose([Resize(config.DIMENSIONS['input_net'],
                                                                       config.DIMENSIONS['input_net']),
                                                                RandomTranslating(),
                                                                RandomScaling(),
                                                                ToTensor(self.input_type),
                                                                RandomRotation()]),
                                  input_type=self.input_type)

        self.set_split_lens = [int(len(self.dataset) * 0.80), int(len(self.dataset) * 0.20)]
        self.train_set, self.val_set = torch.utils.data.random_split(self.dataset, self.set_split_lens)

        self.train_loader = DataLoader(self.train_set,
                                       # sampler=MissingTeethSampler(self.train_set),
                                       batch_size=self.batch_size,
                                       shuffle=True,  # usage of sampler and shuffle is mutually exclusive
                                       num_workers=8,
                                       pin_memory=False)

        self.val_loader = DataLoader(self.val_set,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     num_workers=8,
                                     pin_memory=False)

        print(f'Dataset length: {len(self.dataset)}.')
        print(f'Train dataset length: {len(self.train_set)}.')
        print(f'Validation dataset length: {len(self.val_set)}.')

    def training(self):
        """Training loop. Losses are accumulated and after last epoch, the network state is saved into a file."""
        start_time = time.time()

        print('Training starts...')

        for epoch in range(1, self.epochs + 1):
            print('Epoch {}/{}'.format(epoch, self.epochs))
            print('-' * 10)

            self.running_train_loss = 0.0

            self.running_val_loss = 0.0

            self.epoch_train(epoch)

            if self.val_set is not None:
                self.epoch_validate(epoch)

            print()
            print('Training running loss: {:.4f}'.format(self.running_train_loss/len(self.train_loader)))
            print('Validation running loss: {:.4f}'.format(self.running_val_loss/len(self.val_loader)))
            print('-------------------------------------------------------------')
            self.training_loss_list.append(self.running_train_loss/len(self.train_loader))
            self.validation_loss_list.append(self.running_val_loss/len(self.val_loader))
            print()

            if self.stop_flag:
                break

        self.network.load_state_dict(torch.load('checkpoint.pt'))
        torch.save(self.network.state_dict(), 'UNet-Sampled.pt')  # Rename the resulting saved network after saving
        time_elapsed = time.time() - start_time

        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def epoch_train(self, epoch_id):
        """One epoch of training."""

        self.network.train()

        loop = tqdm(self.train_loader, total=len(self.train_loader))

        for i_batch, sample in enumerate(loop):
            images = sample['image']
            landmarks = sample['landmarks']

            images = images.type(torch.FloatTensor)
            landmarks = landmarks.type(torch.FloatTensor)

            images = images.to(self.device)
            landmarks = landmarks.to(self.device)

            '''Forward.'''
            with torch.cuda.amp.autocast():
                predictions = self.network(images)
                loss = self.criterion(predictions, landmarks, sample['label'])  # Find the loss for the current step

            '''Backward.'''
            # self.scaler.scale(loss).backward()  # Calculate the gradients
            self.scaler.scale(loss / 4).backward()  # Calculate the gradients

            if (i_batch + 1) % 4 == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            # self.optimizer.step()  # Parameters update
                self.optimizer.zero_grad()  # Clear all the gradients before calculating them

            self.running_train_loss += loss.item() * images.shape[0]

            loop.set_description('Training Epoch {}/{}'.format(epoch_id, config.HYPERPARAMETERS['epochs']))
            loop.set_postfix(loss=loss.item() * config.HYPERPARAMETERS['batch_size'])

    def epoch_validate(self, epoch_id):
        """One validation epoch."""

        self.network.eval()

        loop = tqdm(self.val_loader, total=len(self.val_loader))
        losses_for_scheduler = []

        with torch.no_grad():
            for i_batch, sample in enumerate(loop):
                images = sample['image']
                landmarks = sample['landmarks']

                images = images.type(torch.FloatTensor)
                landmarks = landmarks.type(torch.FloatTensor)

                images = images.to(self.device)
                landmarks = landmarks.to(self.device)

                predictions = self.network(images)
                loss = self.criterion(predictions, landmarks, sample['label'])

                self.running_val_loss += loss.item() * images.shape[0]
                losses_for_scheduler.append(loss.item() * images.shape[0])

                loop.set_description('Validating Epoch {}/{}'.format(epoch_id, config.HYPERPARAMETERS['epochs']))
                loop.set_postfix(loss=loss.item() * config.HYPERPARAMETERS['batch_size'])

            mean_loss = sum(losses_for_scheduler)/len(losses_for_scheduler)
            self.scheduler.step(mean_loss)

            plt.title(f'Prediction validation - epoch {epoch_id}')
            plt.imshow(predictions[0][12].cpu().detach().numpy())
            plt.show()

            self.early_stopping(mean_loss, self.network)

            if self.early_stopping.early_stop:
                print('Early stopping')
                self.stop_flag = True
