import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

writer.add_scalar("Loss/train", 733.7712, 1)
writer.add_scalar("Loss/train", 540.2130, 2)
writer.add_scalar("Loss/train", 474.2035, 3)
writer.add_scalar("Loss/train", 432.9407, 4)
writer.add_scalar("Loss/train", 404.2678, 5)
writer.add_scalar("Loss/train", 382.2998, 6)
writer.add_scalar("Loss/train", 364.3024, 7)
writer.flush()

