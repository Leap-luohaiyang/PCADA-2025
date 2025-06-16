import torch
from datasets.data_pre import *
from torch.utils.data import Dataset, DataLoader


def send_to_device(tensor, device):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.

    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to send to a given device.
        device (:obj:`torch.device`):
            The device to send the data to

    Returns:
        The same data structure as :obj:`tensor` with all tensors sent to the proper device.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)


class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""
    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        return data

    def __len__(self):
        return len(self.data_loader)


def get_loader(dataset, bs, shu=False, dl=False):
    loader = torch.utils.data.DataLoader(dataset, batch_size=min(bs, len(dataset)),
                                         shuffle=shu, drop_last=dl)
    return loader


def get_data(cfg):
    if cfg.DATASET.NAME == 'pavia':
        data_path_s = 'data/Pavia/paviaU.mat'
        label_path_s = 'data/Pavia/paviaU_gt_7.mat'
        source_data, source_label = load_data_pavia(data_path_s, label_path_s)

        data_path_t = 'data/Pavia/pavia.mat'
        label_path_t = 'data/Pavia/pavia_gt_7.mat'
        target_data, target_label = load_data_pavia(data_path_t, label_path_t)

    if cfg.DATASET.NAME == 'houston':
        data_path_s = 'data/Houston/Houston13.mat'
        label_path_s = 'data/Houston/Houston13_7gt.mat'
        source_data, source_label = load_data_houston(data_path_s, label_path_s)

        data_path_t = 'data/Houston/Houston18.mat'
        label_path_t = 'data/Houston/Houston18_7gt.mat'
        target_data, target_label = load_data_houston(data_path_t, label_path_t)

    if cfg.DATASET.NAME == 'shanghai-hangzhou':
        file_path = 'data/Shanghai-Hangzhou/DataCube.mat'
        source_data, target_data, source_label, target_label = load_data_sh_hz(file_path)

    return source_data, target_data, source_label, target_label
