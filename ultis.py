import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os.path as osp
import pickle
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from sklearn.metrics import confusion_matrix
from moabb.datasets import BNCI2014001, Cho2017, PhysionetMI
from moabb.paradigms import MotorImagery
from beetl.task_datasets import BeetlSleepLeaderboard, BeetlMILeaderboard
import os


def infoSleepData():
    with open("./SleepSource/headerInfo.npy", 'rb') as f:
        info = pickle.load(f)
    print(info)


def getSleepData(numSubject=38):
    X_source, y_source = [], []
    savebase = "./SleepSource/"
    for i_train in range(numSubject):
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


def getSleepTestData():
    X_source = []
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


def dataDistribution(Ys, titleData=""):
    # plot bar chart the number sample for each label
    (unique, counts) = np.unique(np.asarray(Ys), return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print(frequencies)
    plt.title("Bar chart number sample for each label" + titleData)
    plt.bar(unique, counts)
    plt.show()


def balanceData(Xs, ys):
    # trim data
    # TODO
    marker = np.ones([len(ys)])
    label2 = np.where(ys == 2)[0]
    randomRemove2 = np.random.choice(label2, int(len(label2) * 2 / 3), replace=False)
    marker[randomRemove2] = 0

    label0 = np.where(ys == 0)[0]
    randomRemove0 = np.random.choice(label0, int(len(label0) / 2), replace=False)
    marker[randomRemove0] = 0

    newX_source = Xs[np.where(marker == 1)]
    newy_source = ys[np.where(marker == 1)]
    return newX_source, newy_source


class EEG_data(Dataset):
    def __init__(self, datas, targets=None,
                 train=True):

        self.y = targets
        mean = np.mean(datas, axis=1, keepdims=True)
        std = np.std(datas, axis=1, keepdims=True)
        self.X = (datas - mean) / std
        self.X = self.X.astype(np.double)
        self.transform = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return torch.tensor(self.X[idx]), self.y[idx]
        else:
            return torch.tensor(self.X[idx])


def TrainTestLoader(data, testSize=0.1):
    if len(data) == 2:
        X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=testSize, random_state=42)
    else:
        [X_train, y_train, X_test, y_test] = data
    batch_size = 32

    train_dataset = EEG_data(X_train, y_train)
    test_dataset = EEG_data(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)

    return train_loader, test_loader


def plotLossAcc(loss, acc):
    t = [x for x in range(len(loss))]
    plt.plot(t, loss, 'red')
    t = [x for x in range(len(acc))]
    plt.plot(t, acc, 'blue')
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
    plt.show()
    return ax


def chunk(matrix, step_size=128, window_size=128):
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


def chunk_matrix(list_data, list_target, step_size=32, window_size=128):
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
        tmpTarget = [0] * 12
        matrix = np.copy(data[idx])
        noise = np.random.normal(0, 0.1, size=matrix.shape)
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
        lenRandom = matrix.shape[0]
        numRandom = int(lenRandom / 10)
        listFrame = []
        for id in range(numRandom):
            tmp = np.random.randint(1, matrix.shape[0])
            listFrame.append(tmp)
        for f in listFrame:
            if f > 1 and f < lenRandom - 1:
                matrix[f] = matrix[f - 1] + matrix[f + 1] / 2
        # print(matrix.shape)
        list_newdata.append(matrix)
        list_newtarget.append(target[idx])
    return list_newdata, list_newtarget


def randomSwapSample(data, target):
    list_newdata = []
    list_newtarget = []
    for idx in range(len(data)):
        matrix = np.copy(data[idx])
        lenRandom = matrix.shape[0]
        numRandom = 8
        listFrame = []
        for id in range(numRandom):
            tmp = np.random.randint(3, lenRandom - 10)
            listFrame.append(tmp)
        listFrame = np.sort(listFrame)
        for idy, v in enumerate(listFrame):
            if idy > 0 and listFrame[idy] < listFrame[idy - 1]:
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
        listFrame = np.append(listFrame, lenRandom)
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


def augmentData_Swap(Xs, Ys, labels):
    newXs = []
    newYs = []
    for label in labels:
        X_source = Xs[np.where(Ys == label)]
        y_source = Ys[np.where(Ys == label)]
        dataRemove, targetRemove = randomRemoveSample(X_source, y_source)
        newXs.extend(dataRemove)
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


def relabel(l):
    if l == 'left_hand':
        return 0
    elif l == 'right_hand':
        return 1
    else:
        return 2


def trainMIData(ds_src1, ds_src2, ds_tgt, prgm_2classes, prgm_4classes):
    mysubjects = [x + 1 for x in range(9)]
    ss = [x for x in range(9, 30)]
    ss.extend(mysubjects)
    X_src1, label_src1, m_src1 = prgm_2classes.get_data(dataset=ds_src1, subjects=ss)
    X_src1 = np.transpose(X_src1, (0, 2, 1))
    X_src2, label_src2, m_src2 = prgm_4classes.get_data(dataset=ds_src2, subjects=ss)
    X_src2 = np.transpose(X_src2, (0, 2, 1))
    X_tgt, label_tgt, m_tgt = prgm_4classes.get_data(dataset=ds_tgt, subjects=mysubjects)
    X_tgt = np.transpose(X_tgt, (0, 2, 1))

    print("First source dataset has {} trials with {} electrodes and {} time samples".format(*X_src1.shape))
    print("Second source dataset has {} trials with {} electrodes and {} time samples".format(*X_src2.shape))
    print("Third source dataset has {} trials with {} electrodes and {} time samples".format(*X_tgt.shape))

    y_src1 = np.array([relabel(l) for l in label_src1])
    y_src2 = np.array([relabel(l) for l in label_src2])
    y_tgt = np.array([relabel(l) for l in label_tgt])

    window_size = 300

    X_train = np.concatenate((X_src1[:, :window_size, :], X_src2[:, :window_size, :], X_tgt[:, :window_size, :]))
    y_train = np.concatenate((y_src1, y_src2, y_tgt[:]))

    print("Train:  there are {} trials with {} time samples and {} electrodes".format(*X_train.shape))
    return X_train, y_train


def PhyData(ds_src2, ds_tgt, prgm_4classes):
    mysubjects = [x + 1 for x in range(9)]
    ss = [x for x in range(9, 32)]
    ss.extend(mysubjects)
    X_src2, label_src2, m_src2 = prgm_4classes.get_data(dataset=ds_src2, subjects=ss)
    X_src2 = np.transpose(X_src2, (0, 2, 1))
    X_tgt, label_tgt, m_tgt = prgm_4classes.get_data(dataset=ds_tgt, subjects=mysubjects)
    X_tgt = np.transpose(X_tgt, (0, 2, 1))

    print("Second source dataset has {} trials with {} electrodes and {} time samples".format(*X_src2.shape))
    print("Third source dataset has {} trials with {} electrodes and {} time samples".format(*X_tgt.shape))

    y_src2 = np.array([relabel(l) for l in label_src2])
    y_tgt = np.array([relabel(l) for l in label_tgt])

    window_size = 300

    X_train = np.concatenate((X_src2[:, :window_size, :], X_tgt[:, :window_size, :]))
    y_train = np.concatenate((y_src2, y_tgt[:]))

    print("Train:  there are {} trials with {} time samples and {} electrodes".format(*X_train.shape))
    return X_train, y_train


def ChoData(ds_src1, prgm_2classes):
    mysubjects = [x + 1 for x in range(30)]
    X_src1, label_src1, m_src1 = prgm_2classes.get_data(dataset=ds_src1, subjects=mysubjects)
    X_src1 = np.transpose(X_src1, (0, 2, 1))

    print("First source dataset has {} trials with {} electrodes and {} time samples".format(*X_src1.shape))

    y_src1 = np.array([relabel(l) for l in label_src1])

    window_size = 300

    X_train = np.asarray(X_src1[:, :window_size, :])
    y_train = np.asarray((y_src1))

    print("Train:  there are {} trials with {} time samples and {} electrodes".format(*X_train.shape))
    return X_train, y_train


def tranferData_task2(ds_src1, ds_src2, ds_tgt, prgm_2classes, prgm_4classes):
    s1 = [9, 10, 11, 12, 13, 15, 16, 17, 18, 19]
    X_src1, label_src1, m_src1 = prgm_2classes.get_data(dataset=ds_src1, subjects=s1)
    X_src1 = np.transpose(X_src1, (0, 2, 1))
    s2 = [9, 10, 11, 12, 13, 15, 16, 17, 18, 19]
    X_src2, label_src2, m_src2 = prgm_4classes.get_data(dataset=ds_src2, subjects=s2)
    X_src2 = np.transpose(X_src2, (0, 2, 1))
    s3 = [9]
    X_tgt, label_tgt, m_tgt = prgm_4classes.get_data(dataset=ds_tgt, subjects=s3)
    X_tgt = np.transpose(X_tgt, (0, 2, 1))

    print("First source dataset has {} trials with {} electrodes and {} time samples".format(*X_src1.shape))
    print("Second source dataset has {} trials with {} electrodes and {} time samples".format(*X_src2.shape))
    print("Third Source dataset has {} trials with {} electrodes and {} time samples".format(*X_tgt.shape))

    y_src1 = np.array([relabel(l) for l in label_src1])
    y_src2 = np.array([relabel(l) for l in label_src2])
    y_tgt = np.array([relabel(l) for l in label_tgt])

    window_size = 300
    Xs = np.concatenate((X_src1[:, :window_size, :], X_src2[:, :window_size, :], X_tgt[:, :window_size, :]))
    ys = np.concatenate((y_src1, y_src2, y_tgt))

    print("Train:  there are {} trials with {} time samples and {} electrodes".format(*Xs.shape))
    return Xs, ys


def loadTarget_task2():
    _, _, X_MIA_test = BeetlMILeaderboard().get_data(dataset='A')
    print("MI leaderboard A: There are {} trials with {} electrodes and {} time samples".format(*X_MIA_test.shape))
    _, _, X_MIB_test = BeetlMILeaderboard().get_data(dataset='B')
    print("MI leaderboard B: There are {} trials with {} electrodes and {} time samples".format(*X_MIB_test.shape))
    return X_MIA_test, X_MIB_test


def getMIData():
    ds_src1 = Cho2017()
    ds_src2 = PhysionetMI()
    ds_tgt = BNCI2014001()

    fmin, fmax = 0, 60
    # raw = ds_tgt.get_data(subjects=[1])[1]['session_T']['run_1']
    # tgt_channels = raw.pick_types(eeg=True).ch_names

    # print("list channels will be extracted: {}".format(tgt_channels))
    tgt_channels = ['Fz', 'FC1', 'FC2', 'C5', 'C3', 'C1', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1',
                    'Pz', 'P2']
    sfreq = 100
    prgm_2classes = MotorImagery(n_classes=2, channels=tgt_channels, resample=sfreq, fmin=fmin, fmax=fmax)
    prgm_4classes = MotorImagery(n_classes=4, channels=tgt_channels, resample=sfreq, fmin=fmin, fmax=fmax)
    X_train, y_train = trainMIData(ds_src1, ds_src2, ds_tgt, prgm_2classes, prgm_4classes)
    return X_train, y_train


def getPhyData():
    ds_src2 = PhysionetMI()
    ds_tgt = BNCI2014001()

    fmin, fmax = 0, 60
    tgt_channels = ['Fz', 'FC1', 'FC2', 'C5', 'C3', 'C1', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1',
                    'Pz', 'P2']
    sfreq = 100
    prgm_4classes = MotorImagery(n_classes=4, channels=tgt_channels, resample=sfreq, fmin=fmin, fmax=fmax)
    X_train, y_train = PhyData(ds_src2, ds_tgt, prgm_4classes)
    return X_train, y_train


def getChoData(datapath='./preprocData/'):
    # Cho's dataset comprises only data of 2 labels
    ds_src1 = Cho2017()

    fmin, fmax = 0, 60
    tgt_channels = ['Fz', 'FC1', 'FC2', 'C5', 'C3', 'C1', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1',
                    'Pz', 'P2']
    sfreq = 100
    prgm_2classes = MotorImagery(n_classes=2, channels=tgt_channels, resample=sfreq, fmin=fmin, fmax=fmax)
    X_train, y_train = ChoData(ds_src1, prgm_2classes)
    numdata = 3000
    with open(os.path.join(datapath, 'X0data.npy'), 'rb') as f:
        x0data = np.load(f)
    x0data = x0data[:numdata]
    tmp = np.mean([np.min(x0data), np.max(x0data)])
    tmp1 = np.mean([np.min(X_train), np.max(X_train)])
    diff = tmp1 / tmp
    x0data = x0data * diff
    y0data = np.asarray([2] * numdata)
    X_train = np.concatenate([X_train, x0data])
    y_train = np.concatenate([y_train, y0data])
    return X_train, y_train


def mergeAnswer(files):
    f1, f2 = files
    a = np.loadtxt(f1, delimiter=',')
    b = np.loadtxt(f2, delimiter=',')
    c = np.concatenate([a, b])
    return c
