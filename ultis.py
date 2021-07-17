import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os.path as osp
import pickle
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from sklearn.metrics import confusion_matrix


def infoData():
	with open("./SleepSource/headerInfo.npy", 'rb') as f:
		info = pickle.load(f)
	print(info)


def getData():
	X_source, y_source = [], []
	savebase = "./SleepSource/"
	for i_train in range(38):
		with open(osp.join(savebase, "training_s{}r1X.npy".format(i_train)), 'rb') as f:
			subData = pickle.load(f)
			subData = np.transpose(subData, (0, 2, 1))
			X_source.append(subData)
		with open(osp.join(savebase, "training_s{}r1y.npy".format(i_train)), 'rb') as f:
			y_source.append(pickle.load(f))
		with open(osp.join(savebase, "training_s{}r2X.npy".format(i_train)), 'rb') as f:
			subData = pickle.load(f)
			subData = np.transpose(subData, (0, 2, 1))
			X_source.append(subData)
		with open(osp.join(savebase, "training_s{}r2y.npy".format(i_train)), 'rb') as f:
			y_source.append(pickle.load(f))
	X_source = np.concatenate(X_source)
	y_source = np.concatenate(y_source)
	print("Source: there are {} trials with {} time samples and {} electrodes ".format(*X_source.shape))
	return X_source, y_source


def getTargetData():
	X_source= []
	savebase = "./LeaderboardSleep/testing/"
	for i_train in range(6, 18):
		with open(osp.join(savebase, "leaderboard_s{}r1X.npy".format(i_train)), 'rb') as f:
			subData = pickle.load(f)
			subData = np.transpose(subData, (0, 2, 1))
			X_source.append(subData)
		with open(osp.join(savebase, "leaderboard_s{}r2X.npy".format(i_train)), 'rb') as f:
			subData = pickle.load(f)
			subData = np.transpose(subData, (0, 2, 1))
			X_source.append(subData)
	X_source = np.concatenate(X_source)
	print("Source: there are {} trials with {} time samples {} electrodes ".format(*X_source.shape))
	return X_source

def analyzeTrainData(Xs, Ys):
	print("Shape of train data X_source", Xs.shape)
	(unique, counts) = np.unique(np.asarray(Ys), return_counts=True)
	frequencies = np.asarray((unique, counts)).T
	print(frequencies)
	plt.title("chart number of sample for each label")
	plt.bar(unique, counts)
	plt.show()

def balanceData(Xs, ys):
	marker = np.ones([len(ys)])
	label2 = np.where(ys == 2)[0]
	randomRemove2 = np.random.choice(label2, int(len(label2) * 2 / 3 ), replace = False)
	marker[randomRemove2] = 0

	label0 = np.where(ys == 0)[0]
	randomRemove1 = np.random.choice(label0, int(len(label0) / 2), replace = False)
	marker[randomRemove1] = 0

	newX_source = Xs[np.where(marker == 1)]
	newy_source = ys[np.where(marker == 1)]
	return newX_source, newy_source


class EEG_data(Dataset):
	def __init__(self, datas, targets = None, transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]), train = True):

		self.y = targets
		mean = np.mean(datas, axis=1, keepdims=True)
		std = np.std(datas, axis=1, keepdims=True)
		self.X = (datas - mean) / std
		self.X = np.abs(self.X).astype(np.double)*1e3
		self.transform = transforms

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		if self.y is not None:
			return self.transform(self.X[idx]), self.y[idx]
		else:
			return self.transform(self.X[idx])

def TrainTestLoader(Xs, ys, testSize):
	X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=testSize, random_state=42)
	batch_size = 32

	train_dataset = EEG_data(X_train, y_train)
	test_dataset = EEG_data(X_test, y_test)


	train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
		batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size= batch_size)

	return train_loader, test_loader


def plotLossAcc(loss, acc):
	t = [x for x in range(len(llos))]
	plt.plot(t, llos, 'red')
	t = [x for x in range(len(lacc))]
	plt.plot(t, lacc, 'blue')
	plt.show()

def plot_confusion_matrix(y_true, y_pred, classes,
							normalize=False,
							title=None,
							cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if not title:
		if normalize:
			title = 'Normalized confusion matrix'
		else:
			title = 'Confusion matrix, without normalization'

	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	print(cm)
#     # Only use the labels that appear in the data
#     classes = classes[unique_labels(y_true, y_pred)]
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')



	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
		yticks=np.arange(cm.shape[0]),
		# ... and label them with the respective list entries
		xticklabels=classes, yticklabels=classes,
		title=title,
		ylabel='True label',
		xlabel='Predicted label')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
		rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
					ha="center", va="center",
					color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	return ax

