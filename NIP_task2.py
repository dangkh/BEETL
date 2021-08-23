import numpy as np
from ultis import *
import torch.optim as optim
from torch.optim import lr_scheduler
from nets import *
import argparse
import torch
import nets
from modelUltis import *

import random
import os

# Function to seed everything
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train task 2')
	parser.add_argument('-e', type=int, default = 10, help = "Number of training epoch")
	parser.add_argument('-lr', type=float, default = 0.0001, help = "Learning rate")
	parser.add_argument('-v', type=bool, default = False, help = "Visualize data")
	parser.add_argument('-net', type=str, default = "LSTMNet_t2", help = "Model")
	parser.add_argument('-loss', type=str, default = "CrossEntropyLoss", help = "Loss for model")
	parser.add_argument('-optim', type=str, help = "Optimizer for model")
	parser.add_argument('-valid', type=bool, help = "Check on valid fold")
	parser.add_argument('-confusion', type=bool, help = "Confusion matrix")
	parser.add_argument('-save', type=str, help = "Saving model path")
	parser.add_argument('-pretrain', type=str, help = "pretrain model")
	parser.add_argument('-lognum', type=int, default = 100, help = "Log after Number of training epoch")
	parser.add_argument('-data', type=int, default = 0, help = "0: load all data; 1: load Phy and BCI; 2: load Cho")
	parser.add_argument('-datapath', type=str, default = "/mnt/hdd/NIP/BEETL/preprocData/",
						help='Preprocessed data')
	parser.add_argument('-seed', type=int, default = 123,
						help = "Seed")

	args = parser.parse_args()
	seed_everything(args.seed)
	params = vars(args)

	print("Load data ...")
	tmp_file = os.path.join(args.path, 'data.npz')
	if not os.path.exists(tmp_file):
		print("Create tmp data")
		if args.data == 0:
			X_train, y_train = getMIData()
			if params['v'] : dataDistribution(y_train, "y_train")
			X_train, y_train = augmentData(X_train, y_train, labels = [0, 1])
			X_train, y_train = augmentData_Noise(X_train, y_train, labels = [2])
			if params['v'] : dataDistribution(y_train, "y_train")
		elif args.data == 1:
			X_train, y_train = getPhyData()
			if params['v'] : dataDistribution(y_train, "y_train")
			X_train, y_train = augmentData(X_train, y_train, labels = [0, 1])
			X_train, y_train = augmentData_Noise(X_train, y_train, labels = [2])
			if params['v'] : dataDistribution(y_train, "y_train")
		elif args.data == 2:
			X_train, y_train = getChoData(args.datapath)
			if params['v'] : dataDistribution(y_train, "y_train")
			X_train, y_train = augmentData(X_train, y_train, labels = [0, 1, 2])
			if params['v'] : dataDistribution(y_train, "y_train")

		X_train = np.transpose(X_train, (0, 2, 1))
		n_samples, n_channels, n_timestamp = X_train.shape
		X_train = X_train.reshape((n_samples, n_channels, n_timestamp, 1))

		np.savez(tmp_file, X_train=X_train, y_train=y_train)
	else:
		print("Load from: ", tmp_file)
		tmp_data = np.load(tmp_file)

		X_train = tmp_data['X_train']
		y_train = tmp_data['y_train']

	trainLoader, validLoader = TrainTestLoader([X_train, y_train], 0.1)

	print("Train model ...")
	num_class = len(np.unique(y_train))
	model = WvConvNet(3, 28, 10, drop_rate=0.5, flatten=True, stride=1)
	model.double()
	if args.pretrain is not None:
		model.load_state_dict(torch.load(str(args.pretrain)))
	if torch.cuda.is_available():
		model.cuda()
	
	criterion = getattr(nn, params['loss'])()
	lr = params['lr']
	optimizer = getattr(optim, 'Adam')(model.parameters(), lr=lr)
	scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)
	n_epochs = params['e']

	model, llos = trainModel(model, criterion, n_epochs, optimizer, scheduler, trainLoader, validLoader, num_class,
							 params['save'], params['lognum'])

	print("Eval model ...")
