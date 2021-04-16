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
parser.add_argument("--lr", type=float, default=0.002)
parser.add_argument("--resume", type=bool, default=False)
parser.add_argument("--weights", type=str, default="None")

args = parser.parse_args()


data_path = "./ignore/output/"
data_format = ".wav"
get_audio = GetAudio(data_path=data_path, data_format=data_format)


# init tensorboard: !tensorboard --logdir=logs --bind_all
writer = SummaryWriter('ignore/logs/%s' % args.tb)
print("Tensorboard name: %s" % args.tb)


# init model
model = AtssNet(num_blocks=3)
model.cuda()
if args.resume is True:
    model.load_state_dict(torch.load("ignore/weights/%s" % args.weights))

def save_model(epoch):
    torch.save(model.state_dict(), "ignore/weights/checkpoints/checkpoint%i.pth" % epoch)


# init dataloader
train_loader = Loader(batch_size=2, num_workers=8)


# init optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


# loss
criterion = torch.nn.MSELoss(reduction='sum')


def train():
    logger = Logger(len_train=len(train_loader), tb=args.tb)
    for epoch in range(1000):
        logger.init()
        for iter, data in enumerate(train_loader):
            refer_spec, clear_spec, noicy_spec = data
            refer_spec = list(map(lambda item: item.cuda(), refer_spec))
            clear_spec, noicy_spec = clear_spec.cuda(), noicy_spec.cuda()
            pred_spec = model(noicy_spec, refer_spec)
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


if __name__ == "__main__":
    train()

