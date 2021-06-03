import os
import numpy as np
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from dvs_file_reader import DVSFile, SSSPing, Side
from data_annotation import ObjectID


class SSSData(Dataset):
    def __init__(self, dvs_filepath, annotation_dir, side, transform=None):
        self.filename = os.path.split(os.path.normpath(dvs_filepath))[-1]
        self.side = side
        self.dvsfile = DVSFile(dvs_filepath)
        self.sss_pings = self.dvsfile.sss_pings[self.side]
        self.annotation_dir = annotation_dir
        self.annotations = self._load_annotation(self.side)
        self.transform = transform
        self.plot_utils = {
            ObjectID.NADIR.value: {
                'color': 'y',
                'label': 'nadir'
            },
            ObjectID.ROPE.value: {
                'color': 'r',
                'label': 'rope'
            },
            ObjectID.BUOY.value: {
                'color': 'k',
                'label': 'buoy'
            },
        }

    def _load_annotation(self, side):
        """Read side annotations and return a list, where annotation
        of ith ping is found at the ith position of res"""
        annotation_path = os.path.join(
            self.annotation_dir, f'{self.filename}.{side}.objects.annotation')
        res = []
        with open(annotation_path, 'r') as f:
            for line_idx, line in enumerate(f):
                raw_annotation = [int(x) for x in line.split(' ')]

                annotation_lst = []
                for i in range(0, len(raw_annotation), 3):
                    annotation_lst.append([
                        raw_annotation[i], raw_annotation[i + 1],
                        raw_annotation[i + 2]
                    ])
                res.append(annotation_lst)
        return res

    def plot(self, idx):
        data = self[idx]
        plt.figure()
        plt.ylim(0, 1)
        plt.plot(data['data'],
                 linestyle='-',
                 marker='o',
                 markersize=1,
                 linewidth=.5)
        plt.title(f'{self.filename} {self.side} ping {idx}')

        for i, pos in enumerate(data['label'][:, 0]):
            if pos > 0:
                plt.vlines(pos,
                           0,
                           1,
                           colors=[self.plot_utils[i]['color']],
                           label=self.plot_utils[i]['label'])

        plt.legend()
        plt.show()

    def __len__(self):
        return len(self.sss_pings)

    def __getitem__(self, idx):
        """Return a dictionary with the corresponding ping and 1-hot encoded
        labels"""
        ping_padded = np.zeros((1024, 1))
        ping = self.sss_pings[idx].get_ping_array(normalised=True)
        ping_padded[:ping.shape[0], 0] = ping

        annotation = self.annotations[idx]

        label = np.zeros((3, 1))
        for annotation_lst in annotation:
            dim, _, end_idx = annotation_lst
            label[dim] = end_idx

        sample = {
            'data': ping_padded,
            'label': label,
            'label_scaled': label / ping_padded.shape[0]
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor:
    """Convert ndarrays in SSSData items to Tensors"""
    def __call__(self, sample):
        ping, label, label_scaled = sample['data'], sample['label'], sample[
            'label_scaled']

        # numpy ping shape: L x C(1) -> torch ping shape: C(1) x L
        ping = ping.transpose(1, 0)

        # (3, 1) -> (1, 3)
        label = label.transpose(1, 0)
        label_scaled = label_scaled.transpose(1, 0)
        return {
            'data': torch.from_numpy(ping).float(),
            'label': torch.from_numpy(label).float(),
            'label_scaled': torch.from_numpy(label_scaled).float()
        }
