from collections import OrderedDict as edict
import torch

cfg = edict()
cfg.singleModel = True

cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #
cfg.checkpointEC = './checkpoint/EC7.pth.tar'
cfg.checkpointEC_= './checkpoint/EC8.pth.tar'
cfg.checkpointTC = './checkpoint/TC2.pth.tar'
cfg.checkpointDF = './checkpoint/DF3.pth.tar'
cfg.checkpointFC = './checkpoint/FC3.pth.tar'
cfg.checkpointTC_= './checkpoint/TC4.pth.tar'

cfg.filePath = "./test.txt"
cfg.saveFolder = './predict/'
cfg.dataFolder = './testData/'


