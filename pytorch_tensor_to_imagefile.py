from torchvision.utils import save_image

def tensor_to_image(images_tensor, image_name):
  print('debug : images_tensor.shape ', images_tensor.shape) 
  img1 = images_tensor[0]
  save_image(img1, image_name+'.png') 
