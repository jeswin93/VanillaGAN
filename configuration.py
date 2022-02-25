import torch
class Config:
  def __init__(self, batch_size, epochs, nz, k, device, learning_rate):
    self.batch_size = batch_size
    self.epochs = epochs
    self.sample_size = 64
    self.nz = nz
    self.k = k
    self.device = device
    self.learning_rate = learning_rate


config1 = Config(512, 200, 128, 1, 'cuda', 2e-4)
test_config = Config(10, 1, 128, 1, 'cuda', 2e-4)
