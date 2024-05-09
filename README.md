<!-- <p align="center"> -->
<!-- </p> -->
# H2DT 
<a href="https://github.com/unikcc/h2dt">
  <img src="https://img.shields.io/badge/H2DT-0.1-orange" alt="pytorch 1.8.0">
</a>
<a href="https://github.com/unikcc/DiaASQ" rel="nofollow">
  <img src="https://img.shields.io/badge/pytorch-1.9.0-blue" alt="pytorch 1.9.0">
</a>
<a href="https://huggingface.co/docs/transformers/index" rel="nofollow">
  <img src="https://img.shields.io/badge/transformers-4.30.2-cyan" alt="Build Status">
</a>

This repository contains data and code for the AAAI 2024 paper: [Harnessing Holistic Discourse Features and Triadic Interaction for Sentiment Quadruple Extraction in Dialogues](https://ojs.aaai.org/index.php/AAAI/article/view/29807)

------
To clone the repository, please run the following command:

```bash
git clone https://github.com/whulacc/h2dt
```

## Quick Links
- [Overview](#overview)
- [Requirements](#requirements)
- [Code Usage](#code-usage)
- [Citation](#citation)

## Overview
In this paper, we propose a new model, named H2DT, to extract Target-Aspect-Opinion-Sentiment quadruples from the given dialogue.
More details about the task can be found in our [paper](https://ojs.aaai.org/index.php/AAAI/article/view/29807).


<p align="center">
<img src="./data/fig_sample.png" width="50%" />
</p>

## Requirements

This project uses PyTorch for its implementation. Please ensure your system includes the following package versions:

- Python: 3.7+ 
- PyTorch: 1.13.1+

Additional required packages can be installed via pip:
```bash
pip install -r requirements.txt
```
You should install dgl with the following command:
```bash
pip install dgl -f https://data.dgl.ai/wheels/cu117/repo.html
```


## Code Usage

### For Training and Evaluating on the Chinese Dataset
Run the following script to train and evaluate:
```bash
bash scripts/train_zh.sh
```

### For Training and Evaluating on the English Dataset
Run the following script to train and evaluate:
```bash
bash scripts/train_en.sh
```

### GPU Memory Requirements

Below is a table detailing the GPU memory requirements based on dataset and batch size:

| Dataset | Batch Size | GPU Memory Required |
|---------|------------|---------------------|
| Chinese | 2          | 20GB                |
| English | 1          | 20GB                |

### Customizing Hyperparameters
Hyperparameter settings are flexible and can be adjusted within either `main.py` or `src/config.yaml`. Note that configurations in `main.py` will override any settings in `src/config.yaml`.

## Citation
If you would like to cite our work, please use the following format:
```
@inproceedings{lih2dt-aaai-24,
  author       = {Bobo Li and Hao Fei and Lizi Liao and Yu Zhao and Fangfang Su and Fei Li and Donghong Ji},
  title        = {Harnessing Holistic Discourse Features and Triadic Interaction for
                  Sentiment Quadruple Extraction in Dialogues},
  booktitle    = {AAAI},
  pages        = {18462--18470},
  year         = {2024},
}
```
