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

        for dim in range(data['label'].shape[1]):
            pos = np.nonzero(data['label'][:, dim])[0]
            if len(pos) <= 0:
                continue
            plt.vlines([pos.min(), pos.max()],
                       0,
                       1,
                       colors=[self.plot_utils[dim]['color']],
                       label=self.plot_utils[dim]['label'])
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

        label = np.zeros((ping_padded.shape[0], 3))
        for annotation_lst in annotation:
            dim, start_idx, end_idx = annotation_lst

            if dim == ObjectID.NADIR.value:
                label[end_idx, dim] = 1
            else:
                label[end_idx, dim] = 1

        sample = {'data': ping_padded, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor:
    """Convert ndarrays in SSSData items to Tensors"""
    def __call__(self, sample):
        ping, label = sample['data'], sample['label']

        # numpy ping shape: L x C(1) -> torch ping shape: C(1) x L
        ping = ping.transpose(1, 0)
        # numpy label shape: L x C(3) -> torch label shape: C(3) x L
        label = label.transpose(1, 0)
        return {
            'data': torch.from_numpy(ping).float(),
            'label': torch.from_numpy(label).float()
        }


class MockObjects:
    pass
