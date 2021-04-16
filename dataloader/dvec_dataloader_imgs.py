import os, glob, random

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class LibriSpeech300_dvec(Dataset):
    def __init__(self, epoch_len=1e04):
        super().__init__()
        self.speaker_num = 64
        self.utterance_num = 10
        self.epoch_len = epoch_len
        self.data_path = "./audio/aimg/" # "/workspace/db/audio/Libri/LibriSpeech/train-clean-360/"
        self.data_format = "*.png"
        _, self.id_names, _ = next(os.walk(self.data_path))
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
        # crop = 0
        # while spec_tensor[:, :, crop:crop+spec_len].mean() < spec_tensor.mean():
        else: 
            crop = random.randint(0, (spec_tensor.size(2)-spec_len-1))
            return spec_tensor[:, :, crop:crop+spec_len]

    def _get_crop_spec_imgs(self, spec_imgs_list):
        spec_len = random.randint(80, 120)
        id_list = list()
        for id_name in spec_imgs_list:
            utt_list = list()
            for utt_name in id_name:
                spec_img = Image.open(utt_name)
                utt_list.append(self._trans_crop_spec(spec_img, spec_len))
            id_list.append(torch.stack(utt_list, dim=0))
        return torch.stack(id_list, dim=0)

    def __getitem__(self, idx):
        id_names = random.sample(self.id_names, self.speaker_num)
        spec_imgs_list = list(map(lambda item: random.sample(glob.glob(os.path.join(self.data_path, item, self.data_format)), self.utterance_num), id_names))
        spec_tensor = self._get_crop_spec_imgs(spec_imgs_list)
        return spec_tensor


if __name__ == "__main__":
    data = LibriSpeech300_dvec()
    print(data[0].max())