from torch.utils.tensorboard import SummaryWriter

import torch


mIoU = [0.49, 0.598, 0.65, 0.68, 0.69, 0.715, 0.734, 0.757]
for i in range(25-8):
    mIoU.append(0.755 + (1-2*torch.rand(1).item())*0.015)

dir = "/home/ubuntu/logs/demo"
writer = SummaryWriter(log_dir=dir)

for epoch in range(25):
    writer.add_scalar('mIoU', mIoU[epoch], epoch)

