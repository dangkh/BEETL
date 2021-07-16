from ultis import *
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from nets import *
import torch.optim as optim
from torch.optim import lr_scheduler


if __name__ == "__main__":
	infoData()
	Xs, ys = getData()

	SEED=42
	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	np.random.seed(SEED)

	analyzeTrainData(Xs, ys)
	newXs, newYs = balanceData(Xs, ys)
	analyzeTrainData(newXs, newYs)

	trainLoader, testLoader = TrainTestLoader(newXs, newYs, 0.1)

	# train model

	num_class = len(np.unique(ys))
	model = LSTMNet(n_classes=num_class)
	model.double()
	if torch.cuda.is_available():
	  model.cuda()
	criterion = nn.CrossEntropyLoss()
	lr = 1e-4
	optimizer = optim.Adam(model.parameters(), lr=lr)
	scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)
	n_epochs = 100
	llos = []
	lacc = []


	for epoch in range(n_epochs):  # loop over the dataset multiple times
		model.train()
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
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			total_loss += loss.item()
			if i % 200 == 199:    # print every 200 mini-batches
				print('[%d, %5d] loss: %.3f' %
						(epoch + 1, i + 1, running_loss / 200))
				running_loss = 0.0
		mean_loss = total_loss / len(trainLoader)
		llos.append(mean_loss)
		counter = 0
		for idx, data in enumerate(testLoader):
			xx, yy = data
			if torch.cuda.is_available():
				xx = xx.cuda()
				yy = yy.cuda()
			with torch.no_grad():
				model.eval()
				pred = model(xx)
				res = torch.argmax(pred, 1)
				for i, ypred in  enumerate(res):
					if ypred == yy[i].item():
						counter += 1
		acc = counter / len(testLoader)
		lacc.append(acc)
	print('Finished Training')


	# plot info
	counter = 0
	total = 0
	for idx, data in enumerate(testLoader):
		xx, yy = data
		total += len(yy)
		# cuda
		if torch.cuda.is_available():
			xx = xx.cuda()
		with torch.no_grad():
			model.eval()
			pred = model(xx)
			res = torch.argmax(pred, 1)
			for id, ypred in enumerate(res):
				if ypred == yy[id].item():
					counter += 1
	print('acc: {:1f}%'.format(100 * counter / total))

