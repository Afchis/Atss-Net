import torch


class Logger():
    def __init__(self, len_train, tb):
        self.tb = tb
        self.epoch = 0
        self.iter = 0
        self.len_train = len_train

    def init(self):
        self.epoch += 1
        self.disp = {
            "train_iter" : 0,
            "train_loss" : 0,
            "iter_loss" : 0,
            # "time" : []
        }

    def update(self, key, x):
        if key == "train_iter":
            self.iter += 1
            self.disp[key] += 1
        # elif key == "sim":
        #     self.disp[key] = x
        elif key == "iter_loss":
            self.disp[key] = x
        # elif key == "time":
        #     self.disp[key] = x
        else: 
            self.disp[key] += x

    def printer_train(self):
        print(" "*70, end="\r")
        print("Train prosess: [%0.2f" % (100*self.disp["train_iter"]/self.len_train) + chr(37) + "]", "Iter: %s" % self.iter,
              "Loss: %0.3f" % (self.disp["train_loss"]/self.disp["train_iter"]), end="\r")#  "times: %0.1f, %0.1f"% (self.disp["time"][0], self.disp["time"][1]),     "sim+-: %0.1f, %0.1f" % (self.disp["sim"][0], self.disp["sim"][1]),

    def printer_epoch(self):
        head = "Epoch %s" % self.epoch
        print(" "*70, end="\r")
        print(head, "train:",
              "Loss: %0.2f" % (self.disp["train_loss"]/self.disp["train_iter"]))

    def tensorboard_iter(self, writer):
        if self.tb != "None":
            loss = self.disp["iter_loss"]
            writer.add_scalars("%s" % self.tb, {"loss_iter" : loss}, self.iter)

    def tensorboard_epoch(self, writer):
        if self.tb != "None":
            loss = self.disp["train_loss"]/self.disp["train_iter"]
            writer.add_scalars("%s" % self.tb, {"loss" : loss}, self.iter)