import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014001, Cho2017, PhysionetMI
from moabb.paradigms import MotorImagery
from ultis import *
import torch.optim as optim
from torch.optim import lr_scheduler
from nets import *



newModel = LSTMNet_t2(n_classes=3)
newModel.double()
newModel.load_state_dict(torch.load("model_state_dict_task2.pt"))
if torch.cuda.is_available():
  newModel.cuda()

with open('./testfile.npy', 'rb') as f:
    testfile = np.load(f)
print(testfile.shape)

finalTest_dataset = EEG_data(testfile)
finalTest_loader_simple = torch.utils.data.DataLoader(dataset=finalTest_dataset, batch_size= 32)


counter = 0
resTest = []
for idx, data in enumerate(finalTest_loader_simple):
  xx = data
  # cuda
  if torch.cuda.is_available():
    xx = xx.cuda()
  with torch.no_grad():
      newModel.eval()
      pred = newModel(xx)
      res = torch.argmax(pred, 1)
      resTest.extend(res.cpu().numpy())
print(len(resTest))

np.savetxt("answer.txt",[resTest],delimiter=',', fmt = "%d")