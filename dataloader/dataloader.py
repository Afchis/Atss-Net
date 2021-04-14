import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.get_audio import GetAudio


class LibriSpeech300_train(Dataset):
    def __init__(self, epoch_len=50000):
        super().__init__()
        self.epoch_len = epoch_len
        self.data_path = "/workspace/db/audio/Libri/LibriSpeech/train-clean-360/"
        self.get_audio = GetAudio(self.data_path)
        self.trans = transforms.ToTensor()

    def __len__(self):
        return int(self.epoch_len)

    def __getitem__(self, idx):
        refer_spec, clear_spec, noicy_spec = self.get_audio.train_data()
        refer_spec, clear_spec, noicy_spec = self.trans(refer_spec), self.trans(clear_spec), self.trans(noicy_spec)
        return refer_spec, clear_spec, noicy_spec


# def time_align(spec_list):
#     time_shapes = list()
#     for item in spec_list:
#         time_shapes.append(item.size(2))
#     time_shape = min(time_shapes)
#     time = random.randrange(time_shape)
#     out = list(map(lambda item: item[:,:,time_shape-1:time_shape+len(item)-1], spec_list))
#     out = torch.stack(out, dim=0)
#     return out


def collate_fn(batch):
    refer_spec, clear_spec, noicy_spec = list(), list(), list()
    for _refer_spec, _clear_spec, _noicy_spec in batch:
        refer_spec.append(_refer_spec.float())
        clear_spec.append(_clear_spec)
        noicy_spec.append(_noicy_spec)
    clear_spec = torch.stack(clear_spec, dim=0)
    noicy_spec = torch.stack(noicy_spec, dim=0)
    return refer_spec, clear_spec.float(), noicy_spec.float()


def Loader(batch_size, num_workers, shuffle=False):
    train_data = LibriSpeech300_train()
    train_loader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              collate_fn=collate_fn,
                              shuffle=False)
    return train_loader


if __name__ == "__main__":
    data = LibriSpeech300_train()
    data[0]