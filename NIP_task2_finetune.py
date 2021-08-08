import numpy as np
from ultis import *
import torch.optim as optim
from torch.optim import lr_scheduler
from nets import *
import nets
import argparse
import torch
from modelUltis import *


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train task 1')
	parser.add_argument('-e', type=int, default = 10, help = "Number of training epoch")
	parser.add_argument('-lr', type=float, default = 0.0001, help = "Learning rate")
	parser.add_argument('-v', type=bool, default = False, help = "Visualize data")
	parser.add_argument('-loss', type=str, default = "CrossEntropyLoss", help = "Loss for model")
	parser.add_argument('-optim', type=str, help = "Optimizer for model")
	parser.add_argument('-confusion', type=bool, help = "Confusion matrix")
	parser.add_argument('-save', type=str, help = "Saving model path")
	parser.add_argument('-lognum', type=int, default = 100, help = "Log after Number of training epoch")
	args = parser.parse_args()
	params = vars(args)

	dataPath = "./preprocData/"
	Xtrain_files = ["trainA.npy", "trainB.npy"]
	ytrain_files = ["y_A.npy", "y_B.npy"]
	len_files = [(0,400), (400, 1000)]
	listAnwPth = []
	# run for seperate dataset(A and B) then merge into the final answer
	for idx in range(len(Xtrain_files)):
		print("Load data ...")
		with open(dataPath + Xtrain_files[idx], 'rb') as f:
			Xtrain = np.load(f)
		print(Xtrain.shape)
		with open(dataPath + ytrain_files[idx], 'rb') as f:
			ytrain = np.load(f)
		print(ytrain.shape)

		if params['v'] : dataDistribution(ytrain, "y_train_tranfer")
		Xtrain, ytrain = augmentData(Xtrain, ytrain, labels = [0, 1, 0, 1, 2])
		Xtrain, ytrain = augmentData_Swap(Xtrain, ytrain, labels = [0, 1, 0, 1, 2])
		# Xtrain, ytrain = augmentData(Xtrain, ytrain, labels = [0, 1, 2])
		if params['v'] : dataDistribution(ytrain, "y_train_tranfer")
		trainLoader, validLoader = TrainTestLoader([Xtrain, ytrain], 0.1)

		print("Train model ...")

		num_class = len(np.unique(ytrain))
		model = nets.LSTMNet_t2(n_classes=num_class)
		model.double()
		model.load_state_dict(torch.load("model.pt"))
		if torch.cuda.is_available():
			model.cuda()
		
		criterion = getattr(nn, params['loss'])()
		lr = params['lr']
		optimizer = getattr(optim, 'Adam')(model.parameters(), lr=lr)
		scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)
		n_epochs = params['e']

		for param in model.parameters():
			param.requires_grad = False
		model.fc.weight.requires_grad = True
		model.fc.bias.requires_grad = True

		trainModel(model, criterion, n_epochs, optimizer, scheduler, trainLoader, params['save'], params['lognum'])

		print("Eval model ...")
		evaluateModel(model, plotConfusion = True, dataLoader = trainLoader, n_class = num_class)
		evaluateModel(model, plotConfusion = True, dataLoader = validLoader, n_class = num_class)


		# torch.save(model.state_dict(), "model_state_dict_task2_transfered.pt")

		with open(dataPath + '/testfile.npy', 'rb') as f:
			testfile = np.load(f)
		print(testfile.shape)
		fstart, fstop = len_files[idx]
		testbyId = np.copy(testfile[fstart: fstop])
		test_data = EEG_data(testbyId)
		testLoader = torch.utils.data.DataLoader(dataset=test_data, batch_size= 32)

		res = testModel(model, testLoader)
		anwPth = "answer"+ str(idx) +".txt"
		listAnwPth.append(anwPth)
		np.savetxt(anwPth, [res], delimiter=',', fmt = "%d")

	c = mergeAnswer(listAnwPth)
	print(len(c))
	np.savetxt("answer.txt", [c], delimiter=',', fmt = "%d")

