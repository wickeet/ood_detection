import torch
import os
import numpy as np
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
import argparse
from utils.utils import collate_features
import openslide

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


# CUDA_VISIBLE_DEVICES=0,1 nohup python code/clam/extract_features.py --data_slide_dir=/data/datasets/CAMELYON16/original/images/ --data_h5_dir=/data/datasets/CAMELYON16/patches_512_preset/coords/ --csv_path=/data/datasets/CAMELYON16/original/train_test.csv --feat_dir=/data/datasets/CAMELYON16/patches_512_preset/features_resnet50_clam/ --batch_size=512 --no_auto_skip --target_patch_size=256 > cam_features.out 2>&1 &
# CUDA_VISIBLE_DEVICES=2,3 nohup python code/clam/extract_features.py --data_slide_dir=/data/datasets/CAMELYON16/original/images/ --data_h5_dir=/data/datasets/CAMELYON16/patches_512_preset/coords/ --csv_path=/data/datasets/CAMELYON16/original/train_test.csv --feat_dir=/data/datasets/CAMELYON16/patches_512_preset/features_resnet50_bt/ --model_name=resnet50_bt --batch_size=512 --no_auto_skip --target_patch_size=512 > cam_features_bt.out 2>&1 &

# CUDA_VISIBLE_DEVICES=2,3 nohup python code/clam/extract_features.py --data_slide_dir=/data/datasets/CAMELYON16/original/images/ --data_h5_dir=/data/datasets/CAMELYON16/patches_256_preset/coords/ --csv_path=/data/datasets/CAMELYON16/original/train_test.csv --feat_dir=/data/datasets/CAMELYON16/patches_256_preset/features_resnet50_clam/ --model_name=resnet50_clam --batch_size=512 --target_patch_size=256 > cam_features_clam.out 2>&1 &
# CUDA_VISIBLE_DEVICES=2,3 nohup python code/clam/extract_features.py --data_slide_dir=/data/datasets/CAMELYON16/original/images/ --data_h5_dir=/data/datasets/CAMELYON16/patches_256_preset/coords/ --csv_path=/data/datasets/CAMELYON16/original/train_test.csv --feat_dir=/data/datasets/CAMELYON16/patches_256_preset/features_resnet50_bt/ --model_name=resnet50_bt --batch_size=512 --target_patch_size=256 > cam_features_clam.out 2>&1 &

# CUDA_VISIBLE_DEVICES=2,3 nohup python code/clam/extract_features.py --data_slide_dir=/data/datasets/CAMELYON16/original/images/ --data_h5_dir=/data/datasets/CAMELYON16/patches_512_1_preset/coords/ --csv_path=/data/datasets/CAMELYON16/original/train_test.csv --feat_dir=/data/datasets/CAMELYON16/patches_512_1_preset/features_resnet50_bt/ --model_name=resnet50_bt --batch_size=512 --target_patch_size=512 > cam_features_bt.out 2>&1 &

# CUDA_VISIBLE_DEVICES=2,3 nohup python code/clam/extract_features.py --data_slide_dir=/data/datasets/CAMELYON16/original/images/ --data_h5_dir=/data/datasets/CAMELYON16/patches_768_preset/coords/ --csv_path=/data/datasets/CAMELYON16/original/train_test.csv --feat_dir=/data/datasets/CAMELYON16/patches_768_preset/features_resnet50_bt/ --model_name=resnet50_bt --batch_size=512 --target_patch_size=512 > cam_features_bt.out 2>&1 &

# CUDA_VISIBLE_DEVICES=2,3 nohup python code/clam/extract_features.py --data_slide_dir=/data/datasets/CAMELYON16/original/images/ --data_h5_dir=/data/datasets/CAMELYON16/patches_1120_preset/coords/ --csv_path=/data/datasets/CAMELYON16/original/train_test.csv --feat_dir=/data/datasets/CAMELYON16/patches_1120_preset/features_resnet50_bt/ --model_name=resnet50_bt --batch_size=512 --target_patch_size=512 > cam_features_bt.out 2>&1 &

# CUDA_VISIBLE_DEVICES=0,1 nohup python code/clam/extract_features.py --data_slide_dir=/data/datasets/PANDA/PANDA_original/original/train_images/ --data_h5_dir=/data/datasets/PANDA/PANDA_original/patches_512_preset/coords/ --csv_path=/data/datasets/PANDA/PANDA_original/original/train.csv --feat_dir=/data/datasets/PANDA/PANDA_original/patches_512_preset/features_resnet18/ --model_name=resnet18 --batch_size=512 --no_auto_skip --target_patch_size=224 --slide_ext=.tiff > panda_features_512.out 2>&1 &

# CUDA_VISIBLE_DEVICES=0,1 nohup python code/clam/extract_features.py --data_slide_dir=/data/datasets/PANDA/PANDA_original/original/train_images/ --data_h5_dir=/data/datasets/PANDA/PANDA_original/patches_512_preset/coords/ --csv_path=/data/datasets/PANDA/PANDA_original/original/train.csv --feat_dir=/data/datasets/PANDA/PANDA_original/patches_512_preset/features_resnet50/ --model_name=resnet50 --batch_size=512 --target_patch_size=224 --slide_ext=.tiff > panda_features_512.out 2>&1 &

# CUDA_VISIBLE_DEVICES=0,1 nohup python code/clam/extract_features.py --data_slide_dir=/data/datasets/PANDA/PANDA_original/original/train_images/ --data_h5_dir=/data/datasets/PANDA/PANDA_original/patches_512_preset/coords/ --csv_path=/data/datasets/PANDA/PANDA_original/original/train.csv --feat_dir=/data/datasets/PANDA/PANDA_original/patches_512_preset/features_resnet50_bt/ --model_name=resnet50_bt --batch_size=512 --target_patch_size=512 --slide_ext=.tiff > panda_features_512.out 2>&1 &


# CUDA_VISIBLE_DEVICES=0,1 nohup python code/clam/extract_features.py --data_slide_dir=/data/datasets/PANDA/PANDA_original/train_images/ --data_h5_dir=/data/datasets/PANDA/PANDA_original/patches_256_preset/coords/ --csv_path=/data/datasets/PANDA/PANDA_original/train.csv --feat_dir=/data/datasets/PANDA/PANDA_original/patches_256_preset/features_resnet18/ --model_name=resnet18 --batch_size=512 --no_auto_skip --target_patch_size=224 --slide_ext=.tiff > panda_features_256.out 2>&1 &

# CUDA_VISIBLE_DEVICES=0,1 nohup python code/clam/extract_features.py --data_slide_dir=/data/datasets/PANDA/PANDA_original/train_images/ --data_h5_dir=/data/datasets/PANDA/PANDA_original/patches_64_preset/coords/ --csv_path=/data/datasets/PANDA/PANDA_original/train.csv --feat_dir=/data/datasets/PANDA/PANDA_original/patches_64_preset/features_resnet18/ --model_name=resnet18 --batch_size=512 --no_auto_skip --target_patch_size=224 --slide_ext=.tiff > panda_features_64.out 2>&1 &

##############################################

# ferdaous

# CUDA_VISIBLE_DEVICES=0,1 nohup python code/clam/extract_features.py --data_slide_dir=/data/datasets/HGSOC_TCGA/original/slides/ --data_h5_dir=/data/datasets/HGSOC_TCGA/patches_512_preset/coords/ --csv_path=/data/datasets/HGSOC_TCGA/original/slide_labels.csv --feat_dir=/data/datasets/HGSOC_TCGA/patches_512_preset/features_resnet50_bt/ --model_name=resnet50_bt --batch_size=512 --target_patch_size=512 --slide_ext=.svs > hgsoc_features_512.out 2>&1 &

# CUDA_VISIBLE_DEVICES=2,3 nohup python code/clam/extract_features.py --data_slide_dir=/data/datasets/HGSOC_TCGA/original/slides/ --data_h5_dir=/data/datasets/HGSOC_TCGA/patches_512_preset/coords/ --csv_path=/data/datasets/HGSOC_TCGA/original/slide_labels.csv --feat_dir=/data/datasets/HGSOC_TCGA/patches_256_preset/features_resnet50_bt/ --model_name=resnet50_bt --batch_size=512 --target_patch_size=512 --slide_ext=.svs > hgsoc_features_256.out 2>&1 &


###############################################################

# neurips

# CUDA_VISIBLE_DEVICES=0,1 nohup python code/clam/extract_features.py --data_slide_dir=/data/datasets/CAMELYON16/original/images/ --data_h5_dir=/data/datasets/CAMELYON16/patches_512_preset/coords/ --csv_path=/data/datasets/CAMELYON16/original/train_test.csv --feat_dir=/data/datasets/CAMELYON16/patches_512_preset/features_vit_b_32/ --model_name=vit_b_32 --batch_size=512 --no_auto_skip --target_patch_size=512 > cam16_features_vit.out 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python code/clam/extract_features.py --data_slide_dir=/data/datasets/CAMELYON16/original/images/ --data_h5_dir=/data/datasets/CAMELYON16/patches_1120_preset/coords/ --csv_path=/data/datasets/CAMELYON16/original/train_test.csv --feat_dir=/data/datasets/CAMELYON16/patches_1120_preset/features_vit_b_32/ --model_name=vit_b_32 --batch_size=512 --no_auto_skip --target_patch_size=1120 > cam16_features_vit.out 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python code/clam/extract_features.py --data_slide_dir=/data/datasets/CAMELYON16/original/images/ --data_h5_dir=/data/datasets/CAMELYON16/patches_1120_preset/coords/ --csv_path=/data/datasets/CAMELYON16/original/train_test.csv --feat_dir=/data/datasets/CAMELYON16/patches_1120_preset/features_resnet18/ --model_name=resnet18 --batch_size=512 --no_auto_skip --target_patch_size=1120 > cam16_features_resnet18.out 2>&1 &

# Comando válido
# CUDA_VISIBLE_DEVICES=0,1 nohup python src/data/save_patches.py --data_slide_dir=/data/datasets/CAMELYON16/original/images/ --data_h5_dir=/data/datasets/CAMELYON16/patches_256_preset/coords/ --csv_path=/data/datasets/CAMELYON16/original/train_test.csv --batch_size=512 --target_patch_size=256 > cam_features_clam.out 2>&1 &

def compute_w_loader(
        file_path, 
        wsi,
        batch_size=8, 
        verbose=0, 
        print_every=5, 
        custom_downsample=1,
        custom_transforms=None,
        target_patch_size=-1,
        dataset_name='',
        split='train'
    ):


    dataset = Whole_Slide_Bag_FP(
        file_path=file_path, 
        wsi=wsi, 
        pretrained=False, 
        custom_downsample=custom_downsample, 
        custom_transforms=custom_transforms,
        target_patch_size=target_patch_size)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_features)

    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path, len(loader)))

    wsi_name = os.path.splitext(os.path.basename(file_path))[0]
    all_patches = []

    for count, (batch, coords) in enumerate(loader):    
        if count % print_every == 0:
            print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
        batch_np = batch.cpu().numpy()
        all_patches.append(batch_np)
        # Limita el número de parches a 10,000
        if sum([arr.shape[0] for arr in all_patches]) >= 10000:
            break

    # Concatenar todos los parches y limitar a 10,000
    all_patches = np.concatenate(all_patches, axis=0)[:10000]

    patch_dir = os.path.join("dataset", dataset_name, "numpy", split)
    os.makedirs(patch_dir, exist_ok=True)
    patch_path = os.path.join(patch_dir, f"{wsi_name}.npy")
    np.save(patch_path, all_patches)

parser = argparse.ArgumentParser(description='Patch Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default='.tif')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--dataset_name', type=str, default='CAMELYON16')

args = parser.parse_args()

if __name__ == '__main__':

    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    bags_dataset = Dataset_All_Bags(csv_path)
    coords_files = os.listdir(args.data_h5_dir)
    total = len(bags_dataset)

    for bag_candidate_idx in range(total):
        slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, bag_name)
        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)

        print(h5_file_path)
        print(slide_file_path)
        if not os.path.exists(h5_file_path) or not os.path.exists(slide_file_path):
            print('missing {}, skipping'.format(slide_id))
            continue

        if slide_id + '.h5' not in coords_files:
            print(f'No coords for {slide_id}, skipped.')
            continue
        
        if slide_id.startswith('test_'):
            split = 'test'
        else:
            split = 'train'

        time_start = time.time()
        wsi = openslide.open_slide(slide_file_path)
        compute_w_loader(
            h5_file_path,
            wsi,
            batch_size=args.batch_size,
            verbose=1,
            print_every=20,
            custom_downsample=args.custom_downsample,
            custom_transforms=None,
            target_patch_size=args.target_patch_size,
            dataset_name=args.dataset_name,
            split=split
        )
        time_elapsed = time.time() - time_start
        print('\npatch extraction for {} took {} s'.format(slide_id, time_elapsed))

    print('Finished!')