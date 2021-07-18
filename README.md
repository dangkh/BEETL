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
