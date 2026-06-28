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

[ShrimpDiseaseImageBD](https://data.mendeley.com/datasets/jhrtdj9txm) contains annotated shrimp images and has been widely used for shrimp disease detection research.

[TigerShrimpBD](https://data.mendeley.com/datasets/9dj4sk5d55/1) is a 4-class shrimp disease dataset. The four categories contain 978 images of WSSV, 896 images of Yellow Head, 854 images of Black Gill, and 846 images of Healthy shrimp.

[Fish Disease Dataset](https://www.kaggle.com/datasets/subirbiswas19/freshwater-fish-disease-aquaculture-in-south-asia) is a 7-class fish disease dataset.

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
Note: `work_dir/` is generated during training and are not required for running the code.

## Acknowledgement

We thank the authors of the **ShrimpDiseaseImageBD dataset**, **TigerShrimpBD** and the fish disease dataest for making their dataset publicly available to support research on shrimp disease recognition.

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

If you use the publicly available dataset, please cite:

**ShrimpDiseaseBD**
```bibtex
@article{islam2025shrimpdiseasebd,
  title={ShrimpDiseaseBD: An image dataset for detecting shrimp diseases in the aquaculture sector of Bangladesh},
  author={Islam, Md. M. and Sarker, A. and Choudhury, A. and Ahmed, N. and Shafi, A. A. and Niloy, N. T. and Hossain, M. S. and Ali, M. S. and Chowdhury, A. and Ferdaus, Md. H.},
  journal={Data in Brief},
  volume={60},
  pages={111553},
  year={2025},
  doi={10.1016/j.dib.2025.111553},
  url={https://doi.org/10.1016/j.dib.2025.111553}
}
```

**TigerShrimpBD**
```bibtex
@dataset{ahmed2025tigershrimpbd,
  title={TigerShrimpBD: A tiger shrimp image dataset},
  author={Ahmed, S. I. and Farid, D. M.},
  year={2025},
  publisher={Mendeley Data},
  version={V1},
  doi={10.17632/9dj4sk5d55.1},
  url={https://doi.org/10.17632/9dj4sk5d55.1}
}
```

**Fish Disease Dataset**
```bibtex
@misc{subir_biswas_2024,
	title={Freshwater Fish Disease Aquaculture in south asia},
  author={Subir Biswas},
  year={2024},
  publisher={Kaggle},
	doi={10.34740/KAGGLE/DSV/7944185},
  url={https://www.kaggle.com/dsv/7944185},
}
```

## License

This repository is released for research purposes only.
