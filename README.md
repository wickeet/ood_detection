## Setup

### Install
Create a fresh virtualenv (this codebase was developed and tested with Python 3.11) and then install the required packages:

```
pip install -r requirements.txt
```

### Setup paths
Select where you want your data and model outputs stored (if not specified, datasets and models are stored in the directories with the same names within the project)

```
data_root=/root/for/downloaded/dataset
output_root=/root/for/saved/models
```

## Run with DDPM
We'll use the example of MNIST as an in-distribution dataset and [SVHN,CIFAR10, FashionMNIST] as out-of-distribution datasets.
### Download and process datasets (no need for argument)
```bash
python src/data/get_datasets.py --data_root=${data_root}
```

To use your own data, you just need to provide separate csvs containing paths for the train/val/test splits.

### Train models
Examples here use MNIST as the in-distribution dataset (no need for ``--output_dir`` argument).

```bash
python train_ddpm.py \
--output_dir=${output_root} \
--model_name=mnist \
--training_ids=${data_root}/data_splits/MNIST_train.csv \
--validation_ids=${data_root}/data_splits/MNIST_val.csv \
--is_grayscale=1 \
--n_epochs=300 \
--beta_schedule=scaled_linear_beta \
--beta_start=0.0015 \
--beta_end=0.0195
```

You can track experiments in tensorboard
```bash
tensorboard --logdir=${output_root}
```

The code is DistributedDataParallel (DDP) compatible. To train on e.g. 2 GPUs:

```bash
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
train_ddpm.py \
--output_dir=${output_root} \
--model_name=mnist \
--training_ids=${data_root}/data_splits/MNIST_train.csv \
--validation_ids=${data_root}/data_splits/MNIST_val.csv \
--is_grayscale=1 \
--n_epochs=300 \
--beta_schedule=scaled_linear_beta \
--beta_start=0.0015 \
--beta_end=0.0195
```

### Reconstruct data

```bash
python reconstruct.py \
--output_dir=${output_root} \
--model_name=mnist \
--validation_ids=${data_root}/data_splits/MNIST_val.csv \
--in_ids=${data_root}/data_splits/MNIST_test.csv \
--out_ids=${data_root}/data_splits/MNIST_test.csv,${data_root}/data_splits/MNIST_vflip_test.csv,${data_root}/data_splits/MNIST_hflip_test.csv \
--is_grayscale=1 \
--beta_schedule=scaled_linear_beta \
--beta_start=0.0015 \
--beta_end=0.0195 \
--num_inference_steps=100 \
--inference_skip_factor=4 \
--run_val=1 \
--run_in=1 \
--run_out=1
```
The arg `inference_skip_factor` controls the amount of t starting points that are skipped during reconstruction.
This table shows the relationship between values of `inference_skip_factor` and the number of reconstructions, as needed
to reproduce results in Supplementary Table 4 (for max_t=1000).

| **inference_skip_factor:** | 1   | 2   | 3   | 4   | 5   | 8   | 16  | 32  | 64  |
|------------------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| **num_reconstructions:**   | 100 | 50  | 34  | 25  | 20  | 13  | 7   | 4   | 2   |

N.B. For a quicker run, you can choose to only reconstruct a subset of the validation set with e.g. `--first_n_val=1000`
or a subset of the in/out datasets with `--first_n=1000`


### Classify samples as OOD
```bash
python ood_detection.py \
--output_dir=${output_root} \
--model_name=mnist
```

## Citations
```bib
@InProceedings{Graham_2023_CVPR,
    author    = {Graham, Mark S. and Pinaya, Walter H.L. and Tudosiu, Petru-Daniel and Nachev, Parashkev and Ourselin, Sebastien and Cardoso, Jorge},
    title     = {Denoising Diffusion Models for Out-of-Distribution Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {2947-2956}
}
@inproceedings{graham2023unsupervised,
  title={Unsupervised 3D out-of-distribution detection with latent diffusion models},
  author={Graham, Mark S and Pinaya, Walter Hugo Lopez and Wright, Paul and Tudosiu, Petru-Daniel and Mah, Yee H and Teo, James T and J{\"a}ger, H Rolf and Werring, David and Nachev, Parashkev and Ourselin, Sebastien and others},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={446--456},
  year={2023},
  organization={Springer}
}
```
