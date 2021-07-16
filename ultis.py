import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os.path as osp
import pickle
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch


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
	        lenData = subData.shape[0]
	        subData = np.reshape(subData, (lenData, -1, 2))
	        X_source.append(subData)
	    with open(osp.join(savebase, "training_s{}r1y.npy".format(i_train)), 'rb') as f:
	        y_source.append(pickle.load(f))
	X_source = np.concatenate(X_source)
	y_source = np.concatenate(y_source)
	print("Source: there are {} trials with {} electrodes and {} time samples".format(*X_source.shape))
	return X_source, y_source


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
		self.X = datas
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