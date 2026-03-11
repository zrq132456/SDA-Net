# SDA-Net

Official implementation of the paper:

**SDA-Net: An Interpretable Vision-Based Framework for Shrimp Disease Assessment**

This repository is provided to facilitate reproducibility of the results reported in the paper.

Note: This repository contains the minimal implementation required to reproduce the experimental results reported in the paper.

## Environment

Create a Python environment:

```bash
conda create -n sdanet python=3.9
conda activate sdanet
```
Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset used in the paper is **not included** in this repository.

For researchers who wish to test the code, we recommend the following publicly available dataset:

ShrimpDiseaseImageBD Dataset
[https://data.mendeley.com/datasets/jhrtdj9txm](https://data.mendeley.com/datasets/jhrtdj9txm)

This dataset contains annotated shrimp images and has been widely used for shrimp disease detection research.

When preparing your own dataset, please follow the expected format defined in the `dataset/` module.

## Training

Train the model using a configuration file:

```bash
python train.py --config configs/XXX.yaml
```

Example:

```bash
python train.py --config configs/disease_full_itr.yaml
```

Training outputs such as logs, checkpoints, and visualizations will be saved to:

```
work_dir/
```

## Evaluation

Evaluate a trained model:

```bash
python eval.py --config configs/XXX.yaml
```

## Repository Structure

```
SDA-Net
│
├── configs/        # configuration files
├── dataset/        # dataset loading and preprocessing
├── models/         # model architectures
├── utils/          # utility functions
│
├── train.py        # training script
├── eval.py         # evaluation script
│
├── vis/            # visualization outputs
├── work_dir/       # checkpoints and logs (generated during training)
│
├── requirements.txt
└── README.md
```
Note: `vis/` and `work_dir/` are generated during training and are not required for running the code.

## Acknowledgement

We thank the authors of the **ShrimpDiseaseImageBD dataset** for making their dataset publicly available to support research on shrimp disease recognition.

## Citation

If you find this repository useful in your research, please cite:

```bibtex
@article{SDA-Net,
  title={SDA-Net: An Interpretable Vision-Based Framework for Shrimp Disease Assessment},
  author={Anonymous},
  journal={Under review},
  year={2026}
}
```

## License

This repository is released for research purposes only.
