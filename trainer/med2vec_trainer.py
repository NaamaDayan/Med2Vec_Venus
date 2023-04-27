#################################################################
# Code written by Sajad Darabi (sajad.darabi@cs.ucla.edu)
# For bug report, please contact author using the email address
#################################################################
from model.metric import recall_k

import numpy as np
import torch
# from torchvision.utils import make_grid
from base import BaseTrainer
# import torch.multiprocessing as mp
import datetime
import os
from model.med2vec import Med2Vec

# from torch.utils.tensorboard import SummaryWriter


class Med2VecTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, lr_scheduler=None, train_logger=None):
        super(Med2VecTrainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.validation_split = config['data_loader']['args']['validation_split']
        self.do_validation = self.validation_split > 0
        self.lr_scheduler = lr_scheduler
        self.log_step = config['trainer'].get('log_step', int(np.sqrt(data_loader.batch_size)))
        self.shuffle = config['data_loader']['args']['shuffle']
        self.save_dir = config['trainer']['save_dir']
        self.model_args = config['model']['args']

        # creating the indices for the training and validation, the indices will be shuffled each epoch
        # so the validation and training won't be the same
        self.num_batches = int(len(self.data_loader.dataset) // self.data_loader.batch_size)
        # print("num_batches is:",self.num_batches)
        self.num_training_batches = int(self.num_batches*(1- self.validation_split))
        self.train_data_len = self.num_training_batches * self.data_loader.batch_size
        self.val_data_len = (self.num_batches - self.num_training_batches) * self.data_loader.batch_size
        self.batch_indices = np.arange(self.num_batches)
        self.best_val_loss = float('inf')
        self.best_metric_r = 0
        # loss hyperparameter as regularization between the visit loss and code loss
        self.alpha = 0.1






    def _eval_metrics(self, output, target, **kwargs):
        acc_metrics_r, acc_metrics_t = recall_k(output, target,  **kwargs)
        return acc_metrics_r, acc_metrics_t

    def _shuffle_data(self):
        # shuffling the indices of the batches per epoch
        if self.shuffle:
            np.random.shuffle(self.batch_indices)
        self.train_batch_indices = self.batch_indices[:self.num_training_batches]
        self.val_batch_indices = self.batch_indices[self.num_training_batches:]
        # print("train batches indices are:", self.train_batch_indices)
        # print("validation batches indices are:",self.val_batch_indices)
        # print("intersection of batches indices are:", np.intersect1d(self.train_batch_indices,self.val_batch_indices))



    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        """
        self.model.train()
        torch.set_grad_enabled(True)
        total_metrics_r = np.zeros(len(self.metrics))
        total_metrics_t = np.zeros(len(self.metrics))

        total_loss = 0
        batch_count = 0

        for batch_idx in self.train_batch_indices:
            (x, ivec, jvec,mask, d) = self.data_loader.get_batch_data(batch_idx)
            batch_count += 1

            if len(x) < self.data_loader.batch_size:
                continue
            data, ivec, jvec, mask, d = x.to(self.device), ivec.to(self.device), jvec.to(self.device), mask.to(self.device), d.to(self.device)
            self.optimizer.zero_grad()
            probits, emb_w = self.model(data.long())
            loss_dict = self.loss(data, mask.long(), probits, self.model.bce_loss, emb_w, ivec, jvec,
                                  window=self.config["loss_window"])
            # code_loss = self.alpha*loss_dict['code_loss']
            code_loss = loss_dict['code_loss']
            # visit_loss = loss_dict['visit_loss']
            # loss =code_loss + visit_loss
            loss = code_loss


            # if(code_loss.item() > 0.0 and visit_loss.item() >0.0):
            if(code_loss.item() > 0.0 ):
                # print('were good at batch:',batch_count)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            # elif(code_loss.item() > 0.0):
            #     loss = code_loss
            #     loss.backward()
            #     self.optimizer.step()
            #     total_loss += loss.item()
            # else:
            #     loss = visit_loss
            #     loss.backward()
            #     self.optimizer.step()
            #     total_loss += loss.item()
            metrics_r, metrics_t = self._eval_metrics(probits.detach(), data.detach(), mask=mask, k=self.config['metrics'])
            total_metrics_r += metrics_r
            total_metrics_t += metrics_t

            # if self.verbosity >= 2 and (batch_idx % self.log_step == 0):
            #     self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] {}: {:.6f}'.format(
            #         epoch,
            #         batch_count * self.data_loader.batch_size,
            #         self.data_loader.n_samples,
            #         100.0 * batch_count / len(self.data_loader),
            #         'visit_loss', visit_loss,
            #         'code_loss', code_loss))
            if self.verbosity >= 2 and (batch_idx % self.log_step == 0):
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] {}: {:.6f}'.format(
                    epoch,
                    batch_count * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_count / len(self.data_loader),
                    'code_loss', code_loss))

            # print("batch number:", batch_count+1, "code loss is:", code_loss, "visit loss is:", visit_loss)
        total_loss = total_loss / self.train_data_len
        # print("total loss is:",total_loss)
        # total_metrics = total_metrics/ int(self. )
        # print("total metrics are:",total_metrics)
        log = {
            'train_loss': total_loss
            # 'metrics': total_metrics.tolist()
        }
        # getting the validation loss
        val_log = {'val_loss': 0.0,'metrics_r':0.0}
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            if val_log != val_log:
                return {'log': log, 'train_loss': total_loss}

        return {'log': log, 'train_loss': log['train_loss'], 'val_loss': val_log['val_loss'],'metrics_r':val_log['metrics_val']}

    # def _load_best_model(self):
    #     """
    #     Loads the best saved model from a directory.
    #
    #     Args:
    #     model_class (type): The class of the model to load.
    #     save_dir (str): The directory containing the saved model.
    #     device (torch.device): The device to use for computation.
    #
    #     Returns:
    #     model (torch.nn.Module): The loaded model.
    #     """
    #     checkpoint = torch.load(self.save_path)
    #     model = Med2Vec(**self.model_args)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     best_val_loss = checkpoint['best_val_loss']
    #     data_for_testing = checkpoint['validation_data_for_testing']
    #
    #     return model,best_val_loss,data_for_testing

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        """

        self.model.eval()
        total_val_loss = 0
        self.emb_w = None

        batch_count = 0
        with torch.no_grad():
            for batch_idx in self.val_batch_indices:
                (x, ivec, jvec, mask, d) = self.data_loader.get_batch_data(batch_idx)
                batch_count += 1
                if len(x) < self.data_loader.batch_size:
                    continue
                data, ivec, jvec, mask, d = x.to(self.device), ivec.to(self.device), jvec.to(self.device), mask.to(self.device), d.to(self.device)
                probits, self.emb_w = self.model(data.float(), d)
                loss_dict = self.loss(data, mask.long(), probits, self.model.bce_loss, self.emb_w , ivec, jvec,
                                     window=self.config["loss_window"])
                # code_loss = self.alpha * loss_dict['code_loss']
                code_loss =  loss_dict['code_loss']
                # visit_loss = loss_dict['visit_loss']
                # loss = code_loss + visit_loss
                loss = code_loss
                total_val_loss += loss.item()
                # print("batch number:", batch_count + 1, "code loss is:", loss_dict['code_loss'], "visit loss is:",loss_dict['visit_loss'])
            total_val_loss = total_val_loss / self.val_data_len
            metrics_r, metrics_t = self._eval_metrics(probits.detach(), data.detach(), mask=mask,
                                                      k=self.config['metrics'])

            if total_val_loss < self.best_val_loss:
                self.best_val_loss = total_val_loss
                # save_path = os.path.join(self.save_dir, 'best_model.pt')
                self.save_path = './saved/models/best_model.pt'
                # print("new model saved")
                # print("model stat dict is:",self.model.state_dict())
                torch.save({
                    'model_state_dict':  self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'validation_data_for_testing' : self.val_batch_indices,
                    'embeddings':self.emb_w.detach(),
                    'model_args' : self.model_args,
                    'data_loader':self.data_loader
                }, self.save_path)
        return {
            'val_loss': total_val_loss ,'metrics_val' : metrics_r
        }
