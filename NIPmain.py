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
	newXs, newYs = augmentData(Xs, ys, labels = [4])
	newXs, newYs = augmentData_Noise(newXs, newYs, labels = [1])
	newXs, newYs = augmentData_NoiseSwap(newXs, newYs, labels = [3])
	analyzeTrainData(newXs, newYs)
	newXs, newYs = balanceData(newXs, newYs)
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
	n_epochs = 35
	log_batch = 200
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
			if i % log_batch == (log_batch - 1):    # print every 200 mini-batches
				print('[%d, %5d] loss: %.3f' %
						(epoch + 1, i + 1, running_loss / log_batch))
				running_loss = 0.0
		mean_loss = total_loss / len(trainLoader)
		llos.append(mean_loss)
		scheduler.step()
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
	preds = []
	trueLabel = []
	for idx, data in enumerate(testLoader):
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
			for id, ypred in enumerate(res):
				if ypred == yy[id].item():
					counter += 1
	print('acc: {:1f}%'.format(100 * counter / total))
	# plotLossAcc(llos, lacc)
	plot_confusion_matrix(trueLabel, preds, classes=['0', '1', '2', '3', '4', '5'], 
                                      normalize=True, title='Validation confusion matrix')
	plt.show()
	# test the target
	Xtarget = getTargetData()
	finaldataset = EEG_data(Xtarget)
	finalLoader = torch.utils.data.DataLoader(dataset=finaldataset, batch_size= 32)
	counter = 0
	resTest = []
	for idx, data in enumerate(finalLoader):
		xx = data
		# cuda
		if torch.cuda.is_available():
			xx = xx.cuda()
		with torch.no_grad():
			model.eval()
			pred = model(xx)
			res = torch.argmax(pred, 1)
			resTest.extend(res.cpu().numpy())

	np.savetxt("./answer.txt",[resTest],delimiter=',', fmt = "%d")
