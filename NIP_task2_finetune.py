import numpy as np
from ultis import *
import torch.optim as optim
from torch.optim import lr_scheduler
from nets import *
import nets
import argparse
import torch
from modelUltis import *
import random
import NIP_task2


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
    parser = argparse.ArgumentParser(description='Train task 1')
    parser.add_argument('-e', type=int, default=10, help="Number of training epoch")
    parser.add_argument('-lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('-v', type=bool, default=False, help="Visualize data")
    parser.add_argument('-loss', type=str, default="CrossEntropyLoss", help="Loss for model")
    parser.add_argument('-optim', type=str, help="Optimizer for model")
    parser.add_argument('-confusion', type=bool, help="Confusion matrix")
    parser.add_argument('-save', type=str, help="Saving model path")
    parser.add_argument('-lognum', type=int, default=100, help="Log after Number of training epoch")
    parser.add_argument('-pretrain', type=str, help="pretrain model")
    parser.add_argument('-datapath', type=str, default="/mnt/hdd/NIP/BEETL/preprocData/",
                        help='Preprocessed data')
    parser.add_argument('-seed', type=int, default=123,
                        help="Seed")

    args = parser.parse_args()
    seed_everything(args.seed)

    params = vars(args)

    dataPath = args.datapath
    Xtrain_files = ["trainA.npy", "trainB.npy"]
    ytrain_files = ["y_A.npy", "y_B.npy"]
    len_files = [(0, 400), (400, 1000)]
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

        if params['v']: dataDistribution(ytrain, "y_train_tranfer")
        Xtrain, ytrain = augmentData(Xtrain, ytrain, labels=[0, 1, 0, 1, 2])
        Xtrain, ytrain = augmentData_Swap(Xtrain, ytrain, labels=[0, 1, 0, 1, 2])
        # Xtrain, ytrain = augmentData(Xtrain, ytrain, labels = [0, 1, 2])
        if params['v']: dataDistribution(ytrain, "y_train_tranfer")
        Xtrain = np.transpose(Xtrain, (0, 2, 1))
        n_samples, n_channels, n_timestamp = Xtrain.shape
        Xtrain = Xtrain.reshape((n_samples, 1, n_timestamp, n_channels))
        Xtrain = np.transpose(Xtrain, (0, 1, 3, 2))


        trainLoader, validLoader = TrainTestLoader([Xtrain, ytrain], 0.1)

        print("Train model ...")

        num_class = len(np.unique(ytrain))
        model = NIP_task2.get_model()
        if args.pretrain is not None:
            model.load_state_dict(torch.load(args.pretrain))

        if torch.cuda.is_available():
            model.cuda()

        criterion = getattr(nn, params['loss'])()
        lr = params['lr']
        optimizer = getattr(optim, 'Adam')(model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)
        n_epochs = params['e']

        #for param in model.parameters():
        #    param.requires_grad = False
        #model.fc.weight.requires_grad = True
        #model.fc.bias.requires_grad = True

        trainModel(model, criterion, n_epochs, optimizer, scheduler, trainLoader, validLoader, 3, params['save'], params['lognum'])

        print("Eval model ...")
        evaluateModel(model, plotConfusion=True, dataLoader=trainLoader, n_class=num_class)
        evaluateModel(model, plotConfusion=True, dataLoader=validLoader, n_class=num_class)

        # torch.save(model.state_dict(), "model_state_dict_task2_transfered.pt")

        with open(dataPath + '/testfile.npy', 'rb') as f:
            testfile = np.load(f)
        print('Test file: ', testfile.shape)
        fstart, fstop = len_files[idx]
        testbyId = np.copy(testfile[fstart: fstop])
        testbyId = np.transpose(testbyId, (0, 2, 1))
        n_samples, n_channels, n_timestamp = testbyId.shape
        testbyId = testbyId.reshape((n_samples, 1, n_timestamp, n_channels))
        testbyId = np.transpose(testbyId, (0, 1, 3, 2))

        test_data = EEG_data(testbyId)
        testLoader = torch.utils.data.DataLoader(dataset=test_data, batch_size=32)

        res = testModel(model, testLoader)
        anwPth = "answer" + str(idx) + ".txt"
        listAnwPth.append(anwPth)
        print("Predict output: {}, size = {}".format(anwPth, len(res)))
        np.savetxt(anwPth, [res], delimiter=',', fmt="%d")

    c = mergeAnswer(listAnwPth)
    print('Merge len: ', len(c))
    np.savetxt("answer.txt", [c], delimiter=',', fmt="%d")
