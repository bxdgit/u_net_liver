from torch.utils.data import Dataset
from glob import glob
import PIL.Image as Image
import os
import nibabel as nb
import numpy as np


# def make_dataset(root):
#     imgs=[]
#     n=len(os.listdir(root))//2
#     for i in range(n):
#         img=os.path.join(root,"%03d.png"%i)
#         mask=os.path.join(root,"%03d_mask.png"%i)
#         imgs.append((img,mask))
#     return imgs

def make_dataset(root):
    images = sorted(glob(os.path.join(root, "img*.nii.gz")))
    labels = sorted(glob(os.path.join(root, "label*.nii.gz")))
    files = [(img, label) for img, label in zip(images, labels)]
    return files



class LiverDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path,y_path = self.imgs[index]
        img_x = np.array(nb.load(x_path).dataobj,np.float32).squeeze()
        img_y = np.array(nb.load(y_path).dataobj,np.float32).squeeze()
       
        # img_x = Image.open(x_path)
        # img_y = Image.open(y_path)
        #idx = self.imgs[index]
        # img_x = np.array(nb.load(idx['image']).dataobj,np.float32).squeeze()
        # img_y = np.array(nb.load(idx["mask"]).dataobj,np.float32).squeeze()
      
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    print("hello world")
    liver_dataset = LiverDataset("/home/xindong/project/Unet-pytorch/data/imagesTr",transform=None,target_transform=None)
    img_x,img_y = liver_dataset.__getitem__(0)
    print(img_x.shape)
