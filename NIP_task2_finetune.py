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



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train task 1')
	parser.add_argument('-e', type=int, default = 10, help = "Number of training epoch")
	parser.add_argument('-lr', type=float, default = 0.0001, help = "Learning rate")
	parser.add_argument('-v', type=bool, default = False, help = "Visualize data")
	parser.add_argument('-loss', type=str, default = "CrossEntropyLoss", help = "Loss for model")
	parser.add_argument('-optim', type=str, help = "Optimizer for model")
	parser.add_argument('-confusion', type=bool, help = "Confusion matrix")
	parser.add_argument('-save', type=bool,default = False, help = "Save")
	parser.add_argument('-mp', type=str, default = "model.pkl", help = "Saving model path")
	parser.add_argument('-lognum', type=int, default = 100, help = "Number of training epoch")
	args = parser.parse_args()
	params = vars(args)

	Xtrain_files = []
	ytrain_files = []

	print("Load data ...")
	with open('trainfile.npy', 'rb') as f:
		Xtrain = np.load(f)
	print(Xtrain.shape)
	with open('./y_train_tranfer.npy', 'rb') as f:
		y_train = np.load(f)
	print(y_train.shape)

	if params['v'] : dataDistribution(y_train, "y_train_tranfer")
	Xtrain, ytrain = augmentData(Xtrain, y_train, labels = [0, 1, 2])
	trainLoader, validLoader = TrainTestLoader([Xtrain, ytrain], 0.1)

	print("Train model ...")

	num_class = len(np.unique(y_train))
	model = getattr(nets, params['net'])(n_classes=num_class)
	model.double()
	model.load_state_dict(torch.load("model_state_dict_task2.pt"))
	if torch.cuda.is_available():
		model.cuda()
	
	criterion = getattr(nn, params['loss'])()
	lr = params['lr']
	optimizer = getattr(optim, 'Adam')(model.parameters(), lr=lr)
	scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)
	n_epochs = params['e']

	for param in newModel.parameters():
		param.requires_grad = False
	newModel.fc.weight.requires_grad = True
	newModel.fc.bias.requires_grad = True

	trainModel(model, criterion, n_epochs, optimizer, scheduler, trainLoader, params['save'], params['lognum'])

	print("Eval model ...")
	evaluateModel(model, plotConfusion = True, dataLoader = trainLoader, n_classes = num_class)
	evaluateModel(model, plotConfusion = True, dataLoader = validLoader, n_classes = num_class)


	torch.save(model.state_dict(), "model_state_dict_task2_transfered.pt")

	with open('./testfile.npy', 'rb') as f:
		testfile = np.load(f)
	print(testfile.shape)

	test_data = EEG_data(testfile)
	testLoader = torch.utils.data.DataLoader(dataset=test_data, batch_size= 32)

	res = testModel(model, testLoader)

	np.savetxt("answer.txt",[resTest],delimiter=',', fmt = "%d")