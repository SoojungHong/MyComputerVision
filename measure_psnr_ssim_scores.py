from __future__ import print_function
import argparse
import numpy as np
import skimage
import torch
import cv2
import yaml
import os
from torchvision import models, transforms
from torch.autograd import Variable
import shutil
import glob
import tqdm
from util.metrics import PSNR
from albumentations import Compose, CenterCrop, PadIfNeeded
from PIL import Image

#ORG from ssim.ssimlib import SSIM
from skimage.metrics import structural_similarity as SSIM #SJ_FIX
from skimage import measure #SJ_FIX


from models.networks import get_generator


def get_args():
	parser = argparse.ArgumentParser('Test an image')
	parser.add_argument('--img_folder', required=True, help='GoPRO Folder')
	parser.add_argument('--weights_path', required=True, help='Weights path')

	return parser.parse_args()


def prepare_dirs(path):
	if os.path.exists(path):
		shutil.rmtree(path)
	os.makedirs(path)


def get_gt_image(path):
	dir, filename = os.path.split(path)
	base, seq = os.path.split(dir)
	#ORG base, _ = os.path.split(base)
	#ORG img = cv2.cvtColor(cv2.imread(os.path.join(base, 'sharp', seq, filename)), cv2.COLOR_BGR2RGB)
	img = cv2.cvtColor(cv2.imread(os.path.join(base, 'labels', filename)), cv2.COLOR_BGR2RGB)  # SJ_FIX
	return img


def test_image(model, image_path):
	img_transforms = transforms.Compose([
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	])
	size_transform = Compose([ #TODO : Why Size transform to (736, 1280)?
		# ORG PadIfNeeded(736, 1280)
		PadIfNeeded(255, 255)
	])
	#ORG crop = CenterCrop(720, 1280) #TODO : why crop?
	img = cv2.imread(image_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_s = size_transform(image=img)['image']
	img_tensor = torch.from_numpy(np.transpose(img_s / 255, (2, 0, 1)).astype('float32'))
	img_tensor = img_transforms(img_tensor)
	with torch.no_grad():
		#ORG img_tensor = Variable(img_tensor.unsqueeze(0).cuda())
		img_tensor = Variable(img_tensor.unsqueeze(0).cpu()) #SJ_FIX
		result_image = model(img_tensor)
	result_image = result_image[0].cpu().float().numpy()
	result_image = (np.transpose(result_image, (1, 2, 0)) + 1) / 2.0 * 255.0
	#ORG result_image = crop(image=result_image)['image']
	result_image = result_image.astype('uint8')
	# Convert to PIL Image
	pImg = Image.fromarray(result_image, mode='RGB') #SJ_TEST
	pImg.save(image_path+'result.png')
	gt_image = get_gt_image(image_path)
	_, filename = os.path.split(image_path)
	psnr = PSNR(result_image, gt_image)
	pilFake = Image.fromarray(result_image)
	pilReal = Image.fromarray(gt_image)
	#ORG ssim = SSIM(pilFake).cw_ssim_value(pilReal)
	from SSIM_PIL import compare_ssim
	ssim = compare_ssim(pilFake, pilReal)
	return psnr, ssim


def test(model, files):
	psnr = 0
	ssim = 0

	#-------------------------------------------------------------------
	# For all files in the folder, test the image and get PSNR, SSIM
	#-------------------------------------------------------------------
	for file in tqdm.tqdm(files):
		cur_psnr, cur_ssim = test_image(model, file)
		psnr += cur_psnr
		ssim += cur_ssim
	print("PSNR = {}".format(psnr / len(files)))
	print("SSIM = {}".format(ssim / len(files)))


#-------------------
#  main
#-------------------
if __name__ == '__main__':
	args = get_args()
	with open('config/config.yaml') as cfg:
		config = yaml.load(cfg)
	model = get_generator(config['model'])
	#ORG model.load_state_dict(torch.load(args.weights_path)['model'])

	# check map_location whether cpu or gpu #SJ_FIX_START
	if torch.cuda.is_available():
		map_location = lambda storage, loc: storage.cuda()
	else:
		map_location = 'cpu'
	model.load_state_dict(torch.load(args.weights_path, map_location=map_location)['model'])
	#SJ_FIX_END

	#ORG model = model.cuda()
	#ORG filenames = sorted(glob.glob(args.img_folder + '/test' + '/blur/**/*.png', recursive=True))
	filenames = sorted(glob.glob(args.img_folder + '/images' + '**/*.png', recursive=True))
	test(model, filenames)
