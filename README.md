## Introduction

**LUKE** (**L**anguage **U**nderstanding with **K**nowledge-based
**E**mbeddings) is a new pretrained contextualized representation of words and entities based on transformer. It was
proposed in paper
[LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention](https://arxiv.org/abs/2010.01057).

This reposiory is a modified version of the original LUKE repository. The original repository can be
found [here](https://github.com/studio-ousia/luke)

We mainly focus on using LUKE for relation extraction tasks. We implements some debiasing methods for relation
classification tasks including:

* default: the original model
* EntityMask: mask the entity mentions in the input
* DataAug: data augmentation by replacing entity mentions with other entities
* RDrop: regularize the model by feedforward twice
* Focal: focal loss
* DFocal: Debiased Focal Loss
* PoE: Product-of-Expert used to integrate the predictions of target model and bias model
* Debias: Casual Debias Approach proposed by us
* RDataAug: Regularized Debias Approach proposed by us
* MixDebias: debiasing method proposed by us

## Requirements

You should prepare TACRED series datasets before running the code. The TACRED and Re-TACRED datasets can be downloaded
from the LDC links. After downloading the datasets, you should create the softlink to the datasets by running the
following command:

```bash
bash scripts/setup.sh
```

## Usage

### Training

The training script is located in the scripts/train.sh. You can train the model by executing the following command:

```bash
bash scripts/train.sh 0 tacred default
```

In the above command, the first argument is the GPU index, the second argument is the dataset name, and the third
argument is the debiasing method introduced before (default, EntityMask, DataAug, RDrop, Focal, DFocal, PoE, Debias,
RDataAug, MixDebias). There are some super parameters can be adjusted, you should look up the scripts.

### Evaluating

```bssh
bash scripts/eval.sh 0 tacred default test
```

The above command will evaluate the perfomance of relation classification model trained on tacred dataset under default
setting on the test set.

By default the probability generated by model during predict progress will be stored for CoRE and challenge dataset
generation. You can disable the ability by challenge some variables defined in the scripts/eval.sh.

## Citation

If you use LUKE in your work, please cite the
[original paper](https://aclanthology.org/2020.emnlp-main.523/).

```
@inproceedings{yamada-etal-2020-luke,
    title = "{LUKE}: Deep Contextualized Entity Representations with Entity-aware Self-attention",
    author = "Yamada, Ikuya  and
      Asai, Akari  and
      Shindo, Hiroyuki  and
      Takeda, Hideaki  and
      Matsumoto, Yuji",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.523",
    doi = "10.18653/v1/2020.emnlp-main.523",
}
```

For mLUKE, please cite
[this paper](https://aclanthology.org/2022.acl-long.505/).

```
@inproceedings{ri-etal-2022-mluke,
    title = "m{LUKE}: {T}he Power of Entity Representations in Multilingual Pretrained Language Models",
    author = "Ri, Ryokan  and
      Yamada, Ikuya  and
      Tsuruoka, Yoshimasa",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2022",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.505",
}
```
