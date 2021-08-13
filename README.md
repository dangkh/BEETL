# Transfer learning with EEG data - BEETL

# Overview 
- Competions link: https://beetl.ai/introduction
- Ranking link: https://competitions.codalab.org/competitions/33427
- Provided starter-kit: https://github.com/XiaoxiWei/NeurIPS_BEETL

# TODO
- Test transfer learning
- Test task 2 accuracy

# Pipeline

This pipeline construct based on a stardard-few shot learning set up:

- Create base-train/valid/test loader
- Create backbone models/classifying-head models
- Train feature extractor for base-train loader
- Evaluate feature extractor/classify-head models by valid loader
- Create submit

# Task 1: 

## Quang

### Base Model
```
BaseNet = 1 channel input -> 16 conv kernel size from 5 to 80, 32 channels each -> 3 conv layers -> LSTM
```

### Approaches

#### Simple BaseNet
```
BaseNet -> FC
```

Train with all subjects, 80-20 split:
- Train accuracy: ~85%
- Test accuracy: ~82%
- Transfer accuracy: ~74%
- Submission accuracy: 60%-67%

Train with 34 subjects, test with 5 subjects: all accuracies are similar or lower to train with all subjects with 1~2% differences. Submission accuracy is not better.

Train with transfer subjects: transfer accuracy increases to above 80%, best submission score is 68%.

#### 2 combined BaseNet with one tuned by metric learning
```
Input -> BaseNet1 -> (elementwise multiply) -> FC
      -> BaseNet2 
      
Input -> BaseNet2 -> TripletAngularLoss for subject identification
```
Idea: BaseNet2 will act as a regularization and tune BaseNet1 toward subject's personal features. So that it can transfer well to new subjects.

Train with all subjects, 80-20 split:
- Train accuracy: ~88%
- Test accuracy: ~83%
- Transfer accuracy: ~77%
- Submission accuracy: 57%-67%

Train with 34 subjects, test with 5 subjects: all accuracies are similar or lower to train with all subjects with 2~4% differences. Submission accuracy is not better.

Train with transfer subjects: transfer accuracy increase to 83%, best submission score is 67%.

### Regularizations

Tested:
- DropOut 0.1 -> 0.5: makes validation loss more stable. High DropOut hurts performance.
- Random zero input segments: make validation loss more stable.
- Add random noise to input: no significant differences.
- Weighted loss as in [Class-Balanced Loss](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf): 1-2% validation loss improvements.
- [FocalLoss](https://arxiv.org/abs/1708.02002) instead of CrossEntropyLoss: 1-2% validation loss improvements.


# Task 2:
- Train Phy and BCI dataset
- Finetune on Cho dataset
- Finetune last fc layer on datasetA and datasetB
