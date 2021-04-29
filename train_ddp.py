import argparse
import soundfile as sf
import time

import torch
from torch.utils.tensorboard import SummaryWriter

# import class()
from model.model_head import AtssNet
from utils.logger import Logger
from utils.get_audio_VF import GetAudio

# import def()
from dataloader.dataloader import Loader


parser = argparse.ArgumentParser()

parser.add_argument("--tb", type=str, default="None", help="Tensorboard name")
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--resume", type=bool, default=False)
parser.add_argument("--weights", type=str, default="None")

args = parser.parse_args()


data_path = "./ignore/output/"
data_format = ".wav"
get_audio = GetAudio(data_path=data_path, data_format=data_format)


# init tensorboard: !tensorboard --logdir=logs --bind_all
writer = SummaryWriter('ignore/logs/%s' % args.tb)
print("Tensorboard name: %s" % args.tb)


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    device_ids = list(range(torch.cuda.device_count()))
    gpus = len(device_ids)
    print('GPU detected:', device_ids)
else:
    DEVICE = torch.device("cpu")
    print('No GPU. switching to CPU')


