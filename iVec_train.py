import time
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

# import class()
from model.speaker_encoders.speaker_encoder import SpeakerEncoder
from utils.logger import Logger
from dataloader.iVec_dataloader_imgs import LibriSpeech300_iVec


parser = argparse.ArgumentParser()

parser.add_argument("--tb", type=str, default="None", help="Tensorboard name")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--gamma", type=float, default=0.8)
parser.add_argument("--resume", type=bool, default=False)
parser.add_argument("--weights", type=str, default="None")

args = parser.parse_args()


# init tensorboard: !tensorboard --logdir=logs --bind_all
writer = SummaryWriter('ignore/logs/%s' % args.tb)
print("Tensorboard name: %s" % args.tb)

# init models:
model = SpeakerEncoder(num_speakers=921)
model.cuda()
if args.resume is True:
    model.load_state_dict(torch.load("ignore/weights/iVector_encoder/%s" % args.weights))

# init dataloader
train_loader = LibriSpeech300_iVec(batch_size=128, epoch_len=1e03)

# init optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)

# init loss:
criterion = torch.nn.BCEWithLogitsLoss()

def train():
    logger = Logger(len_train=len(train_loader), tb=args.tb)
    for epoch in range(100):
        logger.init()
        model.train()
        for iter in range(len(train_loader)):
            inputs, labels = train_loader[iter]
            inputs, labels = inputs.cuda(), labels.cuda()
            preds = model(inputs)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.update("train_iter", iter)
            logger.update("train_loss", loss.item())
            logger.update("iter_loss", loss.item())
            logger.printer_train()
            logger.tensorboard_iter(writer=writer)
            logger.tensorboard_epoch(writer=writer)
        scheduler.step()
        logger.printer_epoch()
        torch.save(model.state_dict(), "ignore/weights/iVector_encoder/checkpoints/checkpoint%s.pth" % epoch)


if __name__ == "__main__":
    train()