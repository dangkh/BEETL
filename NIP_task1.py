from ultis import *
import argparse
import numpy as np
import torch
from nets import *
import nets
import torch.optim as optim
from torch.optim import lr_scheduler
from modelUltis import *
 
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train task 1')
	parser.add_argument('-e', type=int, default = 10, help = "Number of training epoch")
	parser.add_argument('-lr', type=float, default = 0.0001, help = "Learning rate")
	parser.add_argument('-v', type=bool, default = False, help = "Visualize data")
	parser.add_argument('-net', type=str, default = "LSTMNet", help = "Model")
	parser.add_argument('-loss', type=str, default = "CrossEntropyLoss", help = "Loss for model")
	parser.add_argument('-optim', type=str, help = "Optimizer for model")
	parser.add_argument('-valid', type=bool, help = "Check on valid fold")
	parser.add_argument('-confusion', type=bool, help = "Confusion matrix")
	parser.add_argument('-save', type=bool,default = False, help = "Save")
	parser.add_argument('-mp', type=str, default = "model.pkl", help = "Saving model path")
	parser.add_argument('-info', type=bool, default = False, help = "Show data info")
	parser.add_argument('-lognum', type=int, default = 100, help = "Number of training epoch")
	args = parser.parse_args()
	params = vars(args)
	
	if params['info']: infoSleepData()
	Xs, ys = getSleepData(38)

	SEED=42
	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	np.random.seed(SEED)
	
	dataDistribution(ys)
	newXs, newYs = augmentData(Xs, ys, labels = [4])
	newXs, newYs = augmentData_Noise(newXs, newYs, labels = [1])
	newXs, newYs = augmentData_NoiseSwap(newXs, newYs, labels = [3])
	dataDistribution(newYs)
	newXs, newYs = balanceData(newXs, newYs)
	dataDistribution(newYs)
	trainLoader, testLoader = TrainTestLoader([newXs, newYs], 0.1)

	# train model

	num_class = len(np.unique(ys))
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


	# eval model
	evaluateModel(model, plotConfusion = True, dataLoader = trainLoader)
	evaluateModel(model, plotConfusion = True, dataLoader = testLoader)

	# test the target

	Xtarget = getSleepTestData()
	finaldataset = EEG_data(Xtarget)
	testDataLoader = torch.utils.data.DataLoader(dataset=finaldataset, batch_size= 32)
	res = testModel(model, testDataLoader)
	np.savetxt("./answerTask1.txt",[res],delimiter=',', fmt = "%d")