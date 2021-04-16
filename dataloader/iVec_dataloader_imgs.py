import os, glob, random

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class LibriSpeech300_iVec(Dataset):
    def __init__(self, batch_size, epoch_len=1e04):
        super().__init__()
        self.epoch_len = epoch_len
        self.batch_size = batch_size
        self.data_path = "/workspace/prj/VoiceFilter/audio/aimg/" # "/workspace/db/audio/Libri/LibriSpeech/train-clean-360/"
        self.data_format = "*.png"
        self.data_names = sorted(glob.glob(os.path.join(self.data_path, "*/", self.data_format)))
        self.num_labels = len(sorted(os.listdir(self.data_path)))
        self.labels = torch.diag(torch.ones(self.num_labels))
        self.trans = transforms.ToTensor()

    def __len__(self):
        return int(self.epoch_len)

    def _trans_crop_spec(self, spec_img, spec_len):
        spec_tensor = self.trans(spec_img)
        if spec_tensor.size(2) < spec_len:
            spec_tensor = torch.cat([spec_tensor, torch.zeros([spec_tensor.size(0), spec_tensor.size(1), spec_len-spec_tensor.size(2)])], dim=2)
            return spec_tensor
        elif spec_tensor.size(2) == spec_len:
            return spec_tensor
        else: 
            crop = random.randint(0, (spec_tensor.size(2)-spec_len-1))
            return spec_tensor[:, :, crop:crop+spec_len]

    def _get_crop_spec_imgs(self, batch_img_list, spec_len):
        batch_tensor_list = list()
        for batch_idx in batch_img_list:
            spec_img = Image.open(batch_idx)
            batch_tensor_list.append(self._trans_crop_spec(spec_img, spec_len))
        return torch.stack(batch_tensor_list, dim=0)

    def __getitem__(self, idx):
        spec_len = random.randint(80, 120)
        batch_img_list = random.sample(self.data_names, self.batch_size)
        batch_tensor = self._get_crop_spec_imgs(batch_img_list, spec_len)
        labels_idx = list(map(lambda item: int(item[38:-9]), batch_img_list))
        batch_label = list()
        for idx in labels_idx:
            batch_label.append(self.labels[idx])
        batch_label = torch.stack(batch_label, dim=0)
        return batch_tensor, batch_label


if __name__ == "__main__":
    data = LibriSpeech300_dvec(batch_size=4)
    out = data[0]

