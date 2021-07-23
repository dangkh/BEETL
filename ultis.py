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
	# print("Source: there are {} trials with {} time samples and {} electrodes ".format(*X_source.shape))
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
	randomRemove0 = np.random.choice(label0, int(len(label0) / 2), replace = False)
	marker[randomRemove0] = 0


	newX_source = Xs[np.where(marker == 1)]
	newy_source = ys[np.where(marker == 1)]
	return newX_source, newy_source


class EEG_data(Dataset):
	def __init__(self, datas, targets = None, transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]), train = True):

		self.y = targets
		mean = np.mean(datas, axis=1, keepdims=True)
		std = np.std(datas, axis=1, keepdims=True)
		self.X = (datas - mean) / std
		self.X = self.X.astype(np.double)*1e3
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

def chunk(matrix, step_size = 128, window_size = 128):
    list_matrix = []
    l, r = 0, window_size - 1
    while r <= matrix.shape[0]:
        subMatrix = np.copy(matrix[l:r])
        list_matrix.append(subMatrix)
        l += step_size
        r += step_size
    l, r = matrix.shape[0] - window_size, matrix.shape[0] - 1
    subMatrix = np.abs(np.copy(matrix[l:r]))
    subMatrix = subMatrix.astype(np.double)
    list_matrix.append(subMatrix)
    return list_matrix


def chunk_matrix(list_data, list_target, step_size = 32, window_size = 128):
    list_matries = []
    list_ys = []
    for idx, matrix in enumerate(list_data):
        matries = chunk(matrix)
        list_matries.extend(matries)
        y_matries = [list_target[idx].astype(int)] * len(matries)
        list_ys.extend(y_matries)

    return list_matries, list_ys

def addNoise(data, target):
    list_newdata = []
    list_newtarget = []
    for idx in range(len(data)):
        tmpTarget = [0]*12
        matrix = np.copy(data[idx])
        noise = np.random.normal(0, 0.1, size= matrix.shape)
        newmatrix = matrix + noise
        newmatrix = newmatrix.astype(np.double)
        list_newdata.append(newmatrix)
        # tmpTarget[target[idx]] = 1
        list_newtarget.append(target[idx])
    return list_newdata, list_newtarget

def randomRemoveSample(data, target):
    list_newdata = []
    list_newtarget = []
    for idx in range(len(data)):
        matrix = np.copy(data[idx])
        numRandom = 500
        listFrame = []
        for id in range(numRandom):
          tmp = np.random.randint(1,matrix.shape[0])
          listFrame.append(tmp)
        for f in listFrame:
          if f > 1 and f < 2999:
            matrix[f] = matrix[f-1] + matrix[f+1] / 2
        # print(matrix.shape)
        list_newdata.append(matrix)
        list_newtarget.append(target[idx])
    return list_newdata, list_newtarget

def randomSwapSample(data, target):
    list_newdata = []
    list_newtarget = []
    for idx in range(len(data)):
        matrix = np.copy(data[idx])
        numRandom = 8
        listFrame = []
        for id in range(numRandom):
            tmp = np.random.randint(3,2990)
            listFrame.append(tmp)
        listFrame = np.sort(listFrame)
        for idy, v in  enumerate(listFrame):
            if idy > 0 and listFrame[idy] < listFrame[idy-1]:
                listFrame[idy] += 1

        list_Matrix = []
        for x in range(4):
            l = x * 2
            r = x * 2 + 1
            dl = listFrame[l]
            dr = listFrame[r]
            tmpMatrix = np.copy(matrix[dl:dr])
            list_Matrix.append(tmpMatrix)
        swapMatrix = []
        arr = np.arange(4)
        np.random.shuffle(arr)
        for x in range(4):
            swapMatrix.append(np.copy(list_Matrix[arr[x]]))


        listFrame = np.insert(listFrame, 0, 0)
        listFrame = np.append(listFrame, 3000)
        list_Matrix = []
        for x in range(5):
            l = x * 2
            r = x * 2 + 1
            dl = listFrame[l]
            dr = listFrame[r]
            tmpMatrix = np.copy(matrix[dl:dr])
            list_Matrix.append(tmpMatrix)


        finalList = []
        for x in range(4):
            finalList.append(list_Matrix[x])
            finalList.append(swapMatrix[x])
        finalList.append(list_Matrix[-1])

        finalMatrix = np.vstack(finalList)
        # print(finalMatrix.shape)
        # if finalMatrix.shape[0] == 3000:
        # 	print(listFrame)
        # 	stop
        list_newdata.append(finalMatrix)
        list_newtarget.append(target[idx])
    return list_newdata, list_newtarget

def augmentData(Xs, Ys, labels):
	newXs = []
	newYs = []
	for label in labels:
		X_source = Xs[np.where(Ys == label)]
		y_source = Ys[np.where(Ys == label)]
		datanoise, targetnoise = addNoise(X_source, y_source)
		dataRemove, targetRemove = randomRemoveSample(X_source, y_source)
		dataSwap, targetSwap = randomSwapSample(X_source, y_source)
		newXs.extend(datanoise)
		newXs.extend(dataRemove)
		newXs.extend(dataSwap)
		newYs.extend(targetnoise)
		newYs.extend(targetRemove)
		newYs.extend(targetSwap)
	newXs.extend(Xs)
	newYs.extend(Ys)
	return np.asarray(newXs), np.asarray(newYs)

def augmentData_NoiseSwap(Xs, Ys, labels):
	newXs = []
	newYs = []
	for label in labels:
		X_source = Xs[np.where(Ys == label)]
		y_source = Ys[np.where(Ys == label)]
		datanoise, targetnoise = addNoise(X_source, y_source)
		dataRemove, targetRemove = randomRemoveSample(X_source, y_source)
		newXs.extend(datanoise)
		newXs.extend(dataRemove)
		newYs.extend(targetnoise)
		newYs.extend(targetRemove)
	newXs.extend(Xs)
	newYs.extend(Ys)
	return np.asarray(newXs), np.asarray(newYs)

def augmentData_Noise(Xs, Ys, labels):
	newXs = []
	newYs = []
	for label in labels:
		X_source = Xs[np.where(Ys == label)]
		y_source = Ys[np.where(Ys == label)]
		datanoise, targetnoise = addNoise(X_source, y_source)
		newXs.extend(datanoise)
		newYs.extend(targetnoise)
	newXs.extend(Xs)
	newYs.extend(Ys)
	return np.asarray(newXs), np.asarray(newYs)
