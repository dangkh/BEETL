import numpy as np
import matplotlib.pyplot as plt
import torch
from ultis import *
from nets import *
from modelUltis import *
import sys


def trainModel(model, criterion, n_epochs, optimizer, scheduler, trainLoader, validLoader, n_class, savePath, log_batch):
	llos = []
	best_acc = 0
	for epoch in range(n_epochs):  # loop over the dataset multiple times
		model.train()
		print("")
		print("epoch:  {0} / {1}   ".format(epoch, n_epochs))
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
				# print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / log_batch))
				percent = int(i *50/ len(trainLoader))
				remain = 50 - percent 
				sys.stdout.write("\r[{0}] {1}% loss: {2: 3f}".format('#'*percent + '-'*remain, percent * 2, running_loss / log_batch))
				sys.stdout.flush()
				running_loss = 0.0
		mean_loss = total_loss / len(trainLoader)
		llos.append(mean_loss)
		scheduler.step()
		sys.stdout.write("\r[{0}] {1}% loss: {2: 3f}".format('#'*50, 100, mean_loss))
		sys.stdout.flush()
		acc = evaluateModel(model, plotConfusion = True, dataLoader = validLoader, n_class=n_class)

		if acc > best_acc:
			best_acc = acc
			if savePath is not None:
				torch.save(model.state_dict(), savePath)

	return model, llos

def evaluateModel(model, plotConfusion, dataLoader, n_class):
	counter = 0
	total = 0
	preds = []
	trueLabel = []
	model.eval()
	for idx, data in enumerate(dataLoader):
		xx, yy = data
		trueLabel.extend(yy.numpy())
		total += len(yy)
		if torch.cuda.is_available():
			xx = xx.cuda()
		with torch.no_grad():
			pred = model(xx)
			res = torch.argmax(pred, 1)
			if torch.cuda.is_available():
				res = res.cpu()
			preds.extend(res.numpy())
			for id, ypred in enumerate(res):
				if ypred == yy[id].item():
					counter += 1
	print('acc: {:1f}%'.format(100 * counter / total))
	if plotConfusion:
		plotCl = [str(x) for x in range(n_class)]
		plot_confusion_matrix(trueLabel, preds, classes= plotCl, normalize=True, title='Validation confusion matrix')
		plt.show()

	return 100 * counter / total


def testModel(model, dataLoader ):
	resTest = []
	model.eval()
	for idx, data in enumerate(dataLoader):
		xx = data
		# cuda
		if torch.cuda.is_available():
			xx = xx.cuda()
		with torch.no_grad():
			pred = model(xx)
			res = torch.argmax(pred, 1)
			resTest.extend(res.cpu().numpy())
	return resTest