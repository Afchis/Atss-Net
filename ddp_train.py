import os
import argparse
import soundfile as sf
import time

import torch
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

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


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def train(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    train_loader = Loader(batch_size=6, num_workers=8)

    # create model and move it to GPU with id rank
    model = AtssNet(num_blocks=3).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    logger = Logger(len_train=len(train_loader), tb=args.tb)
    for epoch in range(1000):
        logger.init()
        for iter, data in enumerate(train_loader):
            refer_spec, clear_spec, noicy_spec = data
            refer_spec = list(map(lambda item: item.to(rank), refer_spec))
            clear_spec, noicy_spec = clear_spec.to(rank), noicy_spec.to(rank)
            pred_spec = ddp_model(noicy_spec, refer_spec)
            loss = criterion(pred_spec, clear_spec)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.update("train_iter", iter)
            logger.update("train_loss", loss.item())
            logger.update("iter_loss", loss.item())
            logger.printer_train()
            logger.tensorboard_iter(writer=writer)
            logger.tensorboard_epoch(writer=writer)
        logger.printer_epoch()
        save_model(epoch)
        noicy_audio = get_audio.spec2wav(noicy_spec.detach().cpu().numpy()[0, 0])
        clear_audio = get_audio.spec2wav(clear_spec.detach().cpu().numpy()[0, 0])
        pred_audio = get_audio.spec2wav(pred_spec.detach().cpu().numpy()[0, 0])

        sf.write(data_path + ("small_data/%i_clear" % epoch) + data_format, clear_audio, 16000)
        time.sleep(1)

        sf.write(data_path + ("small_data/%i_pred" % epoch) + data_format, pred_audio, 16000)
        time.sleep(1)
        
        sf.write(data_path + ("small_data/%i_noicy" % epoch) + data_format, noicy_audio, 16000)
        time.sleep(1)
    cleanup()


    


def run(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    run(train, world_size=3)