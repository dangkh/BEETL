

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


with open('trainfile.npy', 'rb') as f:
    trainfile = np.load(f)
print(trainfile.shape)
with open('./y_train_tranfer.npy', 'rb') as f:
    y_train_tranfer = np.load(f)
print(y_train_tranfer.shape)
analyzeTrainData(trainfile, y_train_tranfer)
augmentData((trainfile, y_train_tranfer, labels = [0, 1, 2]
trainLoader, validLoader = TrainTestLoader([trainfile, y_train_tranfer], 0.1)

newModel = LSTMNet_t2(n_classes=3)
newModel.double()
newModel.load_state_dict(torch.load("model_state_dict_task2_finetuned.pt"))
if torch.cuda.is_available():
	newModel.cuda()

for param in newModel.parameters():
	param.requires_grad = False
newModel.fc.weight.requires_grad = True
newModel.fc.bias.requires_grad = True


criterion = nn.CrossEntropyLoss()
lr = 1e-4
optimizer = optim.Adam(newModel.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)
n_epochs = 30
log_batch = 50
llos = []
lacc = []
for epoch in range(n_epochs):  # loop over the dataset multiple times
    newModel.train()
    print("epoch:     ", epoch)
    running_loss = 0.0
    total_loss = 0
    for i, data in enumerate(trainLoader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        labels = labels.type(torch.LongTensor)
        # CUDA
        if torch.cuda.is_available():
          inputs = inputs.cuda()
          labels = labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = newModel(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        total_loss += loss.item()
        if i % log_batch == (log_batch -1):    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / log_batch))
            running_loss = 0.0
    mean_loss = total_loss / len(trainLoader)
    llos.append(mean_loss)
    counter = 0
    for idx, data in enumerate(trainLoader):
      xx, yy = data
      if torch.cuda.is_available():
        xx = xx.cuda()
        yy = yy.cuda()
      with torch.no_grad():
          newModel.eval()
          pred = newModel(xx)
          res = torch.argmax(pred, 1)
          for i, ypred in  enumerate(res):
            if ypred == yy[i].item():
              counter += 1
    acc = counter / len(y_train)
    lacc.append(acc)
print('Finished Training')
plotLossAcc(llos, lacc)

# plot confusion matrix on training test 
counter = 0
total = 0
preds = []
trueLabel = []
for idx, data in enumerate(trainLoader):
  xx, yy = data
  trueLabel.extend(yy.numpy())
  total += len(yy)
  # cuda
  if torch.cuda.is_available():
    xx = xx.cuda()
  with torch.no_grad():
      newModel.eval()
      pred = newModel(xx)
      res = torch.argmax(pred, 1)
      if torch.cuda.is_available():
          res = res.cpu()
          preds.extend(res.numpy())
    #   print(res)
      for id, ypred in enumerate(res):
        if ypred == yy[id].item():
          counter += 1
# print(counter / total, counter, total)    
print('Test Acc: {:1f}%'.format(100 * counter / total))

plot_confusion_matrix(trueLabel, preds, classes=['0', '1', '2'], 
                                      normalize=True, title='Validation confusion matrix')

# plot confusion matrix on valid test 

counter = 0
total = 0
preds = []
trueLabel = []
for idx, data in enumerate(validLoader):
  xx, yy = data
  trueLabel.extend(yy.numpy())
  total += len(yy)
  # cuda
  if torch.cuda.is_available():
    xx = xx.cuda()
  with torch.no_grad():
      model.eval()
      pred = model(xx)
      res = torch.argmax(pred, 1)
      if torch.cuda.is_available():
          res = res.cpu()
          preds.extend(res.numpy())
    #   print(res)
      for id, ypred in enumerate(res):
        if ypred == yy[id].item():
          counter += 1
# print(counter / total, counter, total)    
print('Valid Acc: {:1f}%'.format(100 * counter / total))

plot_confusion_matrix(trueLabel, preds, classes=['0', '1', '2'], 
                                      normalize=True, title='Validation confusion matrix')

torch.save(model.state_dict(), "model_state_dict_task2_transfered.pt")