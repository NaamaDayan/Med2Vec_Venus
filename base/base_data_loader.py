import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import BatchSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.batch_idx = 0
        self.n_samples = len(dataset)
        self.batch_size = batch_size

        # self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
            }
        super(BaseDataLoader, self).__init__(**self.init_kwargs)

    def get_batch_data(self, batch_idx):
        all_x, all_ivec, all_jvec, all_d = [], [], [], []
        for i in range(batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size):
            (x, ivec, jvec, d) = self.dataset.__getitem__(i)
            all_x.append(torch.tensor(x))
            all_ivec.append(torch.tensor(ivec).clone().detach())
            all_jvec.append(torch.tensor(jvec).clone().detach())
            all_d.append(torch.tensor(d).clone().detach())
        ivec_tensor = torch.cat(all_ivec, dim=0)
        jvec_tensor = torch.cat(all_jvec, dim=0)
        mask = [int(torch.sum(t) > 0) for t in all_x]
        mask = torch.tensor(mask)
        all_x = torch.stack(all_x)
        all_d = torch.stack(all_d)
        return all_x, ivec_tensor, jvec_tensor, mask, all_d

    # def _split_sampler(self, split):
    #     if split == 0.0:
    #         return None, None
    #     idx_full = np.arange(self.n_samples)
    #     batch_size = self.batch_size
    #     n_batches = self.n_samples // batch_size
    #
    #     # shuffle indexes of batches only if shuffle is true
    #     # added for med2vec dataset where order matters
    #     if self.shuffle:
    #         np.random.seed(0)
    #         np.random.shuffle(idx_full)
    #
    #     # Split data into train and validation sets
    #     len_valid = int(n_batches * split)
    #     valid_idx = idx_full[:len_valid * batch_size]
    #     train_idx = idx_full[len_valid * batch_size:]
    #
    #     # Shuffle batches if shuffle is true
    #     if self.shuffle:
    #         np.random.shuffle(train_idx)
    #         np.random.shuffle(valid_idx)
    #
    #     # Create sampler for train and validation sets
    #     train_sampler = BatchSampler(SubsetRandomSampler(train_idx), batch_size, drop_last=False)
    #     valid_sampler = BatchSampler(SubsetRandomSampler(valid_idx), batch_size, drop_last=False)
    #
    #     # turn off shuffle option which is mutually exclusive with sampler
    #     self.n_samples = len(train_idx)
    #
    #     return train_sampler, valid_sampler
    # def split_validation(self):
    #     if self.valid_sampler is None:
    #         return None
    #     else:
    #         return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

