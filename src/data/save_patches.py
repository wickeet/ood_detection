import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline as resnet50_clam
import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import timm
import h5py
import openslide

import torchvision
from torchvision.models.resnet import Bottleneck, ResNet

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


class ViTWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ViTWrapper, self).__init__()
        self.model = model

    def __call__(self, x):

        x = self.model._process_input(x)
        n = x.shape[0]

        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat((batch_class_token, x), dim=1)

        x = self.model.encoder(x)
        x = x[:, 0]
        return x

class ResNetTrunk(ResNet):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		del self.fc  # remove FC layer
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)

		x = x.view(x.size(0), -1)

		return x

def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url

def resnet50_ssl(pretrained, progress, key, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        print(verbose)
    return model

def get_ssl_transforms():
	transforms = torchvision.transforms.Compose(
		[
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[ 0.70322989, 0.53606487, 0.66096631 ], std=[ 0.21716536, 0.26081574, 0.20723464 ]),
		]
	)
	return transforms

def build_model(model_name):
	if model_name == 'resnet18':
		weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
		model = torchvision.models.resnet18(weights = weights)
		model = torch.nn.Sequential(*list(model.children())[:-1], torch.nn.Flatten())
		transforms = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])

	elif model_name == 'resnet50':
		weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
		model = torchvision.models.resnet50(weights = weights)
		model = torch.nn.Sequential(*list(model.children())[:-1], torch.nn.Flatten())
		transforms = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
	elif model_name == 'vit_b_32':
		model = torchvision.models.vit_b_32(weights='IMAGENET1K_V1')
		model = ViTWrapper(model)	
		transforms = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Resize((224, 224)),
			torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
	elif model_name == 'resnet50_clam':
		model = resnet50_clam(pretrained=True)
		transforms = None
	elif model_name == 'resnet50_bt':
		model = resnet50_ssl(pretrained=True, progress=False, key="BT")
		transforms = get_ssl_transforms()
	elif model_name == 'resnet50_mocov2':
		model = resnet50_ssl(pretrained=True, progress=False, key="MoCoV2")
		transforms = get_ssl_transforms()
	elif model_name == 'resnet50_swav':
		model = resnet50_ssl(pretrained=True, progress=False, key="SwAV")
		transforms = get_ssl_transforms()
	elif model_name == 'UNI':
		local_dir = "/data/data_fjaviersaezm/models/UNI/assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"
		model = timm.create_model(
			"vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
		)
		model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
		transforms = torchvision.transforms.Compose(
			[
				torchvision.transforms.Resize(224),
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
			]
		)
		model.eval()
	return model, transforms

def compute_w_loader(
		file_path, 
		output_path, 
		wsi, 
		model,
 		batch_size = 8, 
		verbose = 0, 
		print_every=5, 
		pretrained=True, 
		custom_downsample=1,
		custom_transforms=None,
		target_patch_size=-1
	):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	dataset = Whole_Slide_Bag_FP(
		file_path=file_path, 
		wsi=wsi, 
		pretrained=pretrained, 
		custom_downsample=custom_downsample, 
		custom_transforms=custom_transforms,
		target_patch_size=target_patch_size)
	x, y = dataset[0]
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	wsi_name = os.path.splitext(os.path.basename(file_path))[0]

	for count, (batch, coords) in enumerate(loader):	
		if count % print_every == 0:
			print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
		batch_np = batch.cpu().numpy()
		for i in range(batch_np.shape[0]):
			x_coord, y_coord = coords[i]
			patch_name = f"{wsi_name}_{x_coord}_{y_coord}.npy"
			patch_path = os.path.join(wsi_name.upper(), "numpy", patch_name)
			np.save(patch_path, batch_np[i])
	
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.tif')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--model_name', type=str, default='resnet50_clam')

args = parser.parse_args()


if __name__ == '__main__':

	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'npy_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'npy_files'))
	coords_files = os.listdir(args.data_h5_dir)

	print('loading model checkpoint')
	model, custom_transforms = build_model(args.model_name)
	model = model.to(device)
	
	# print_network(model)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		
	model.eval()
	total = len(bags_dataset)

	for bag_candidate_idx in range(total):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		print(h5_file_path)
		print(slide_file_path)
		if not os.path.exists(h5_file_path) or not os.path.exists(slide_file_path):
			print('missing {}, skipping'.format(slide_id))
			continue		

		if not args.no_auto_skip and slide_id+'.npy' in dest_files:
			print('skipped {}'.format(slide_id))
			continue

		if slide_id+'.h5' not in coords_files:
			print(f'No coords for {slide_id}, skipped.')
			continue


		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)
		output_file_path = compute_w_loader(
			h5_file_path, 
			output_path, 
			wsi, 
			model = model, 
			batch_size = args.batch_size,
			verbose = 1, 
			print_every = 20, 
			custom_downsample=args.custom_downsample, 
			custom_transforms=custom_transforms,
			target_patch_size=args.target_patch_size
		)
		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		file = h5py.File(output_file_path, "r")

		features = file['features'][:]
		print('features size: ', features.shape)
		print('coordinates size: ', file['coords'].shape)
		bag_base, _ = os.path.splitext(bag_name)
		np.save(os.path.join(args.feat_dir, 'npy_files', bag_base + '.npy'), features)
		# features = torch.from_numpy(features)
		# torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))
	
	print('Finished!')



