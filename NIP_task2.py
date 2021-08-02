
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
	parser.add_argument('-net', type=str, default = "LSTMNet_t2", help = "Model")
	parser.add_argument('-loss', type=str, default = "CrossEntropyLoss", help = "Loss for model")
	parser.add_argument('-optim', type=str, help = "Optimizer for model")
	parser.add_argument('-valid', type=bool, help = "Check on valid fold")
	parser.add_argument('-confusion', type=bool, help = "Confusion matrix")
	parser.add_argument('-save', type=bool,default = False, help = "Save")
	parser.add_argument('-mp', type=str, default = "model.pkl", help = "Saving model path")
	parser.add_argument('-lognum', type=int, default = 100, help = "Number of training epoch")
	args = parser.parse_args()
	params = vars(args)

	print("Load data ...")
	X_train, y_train, X_val, y_val = getMIData()
	if params['v'] : dataDistribution(y_train, "y_train")
	if params['v'] : dataDistribution(y_val, "y_val")

	trainLoader, validLoader = TrainTestLoader([X_train, y_train, X_val, y_val])

	print("Train model ...")
	num_class = len(np.unique(y_train))
	model = getattr(nets, params['net'])(n_classes=num_class)
	model.double()
	if torch.cuda.is_available():
		model.cuda()
	
	criterion = getattr(nn, params['loss'])()
	lr = params['lr']
	optimizer = getattr(optim, 'Adam')(model.parameters(), lr=lr)
	scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)
	n_epochs = params['e']

	trainModel(model, criterion, n_epochs, optimizer, scheduler, trainLoader, params['save'], params['lognum'])

	print("Eval model ...")
	evaluateModel(model, plotConfusion = True, dataLoader = trainLoader, n_classes = num_class)
	evaluateModel(model, plotConfusion = True, dataLoader = validLoader, n_classes = num_class)

	torch.save(model.state_dict(), "model_state_dict_task2.pt")
