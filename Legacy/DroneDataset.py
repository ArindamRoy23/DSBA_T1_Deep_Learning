from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms as T
import torch

class DroneDataset(Dataset):

    def __init__(self,
                 img_path,
                 mask_path,
                 X,
                 mean,
                 std,
                 transform=None,
                 patch=False):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.patches = patch
        self.mean = mean
        self.std = std
        # self.labels = labels
        # self.fix_dict = {i: labels[i] for i in range(len(labels))}

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.tif')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + self.X[idx] + '.png', cv2.IMREAD_GRAYSCALE)

        # Crop the center of the image
        img = np.asarray(img)
        # height, width = img.shape[0], img.shape[1]
        # new_height, new_width = min(height, width), min(height, width)
        # top = (height - new_height) // 2
        # left = (width - new_width) // 2
        # img = img[top: top + new_height, left: left + new_width, :]

        # Crop the center of the mask
        mask = np.asarray(mask)
        # height, width = mask.shape[0], mask.shape[1]
        # new_height, new_width = min(height, width), min(height, width)
        # top = (height - new_height) // 2
        # left = (width - new_width) // 2
        # mask = mask[top: top + new_height, left: left + new_width]

        # Compute the aspect ratio of the image
        # aspect_ratio = img.shape[1]/img.shape[0]
        # aspect_ratio_mask = mask.shape[1] / mask.shape[0]

        # Define the target size
        target_height = 1024

        # target width as per aspect ration
        # target_width = int(target_height / aspect_ratio)
        # target_width_mask = int(target_height / aspect_ratio_mask)

        # fixed width
        # target_width = int(target_height/1.33)
        # target_width_mask = int(target_height/1.33)

        target_width = 800
        target_width_mask = 800

        target_size = (target_height, target_width)
        target_size_mask = (target_height, target_width_mask)

        # Resize the image
        img = cv2.resize(img, target_size)
        mask = cv2.resize(mask, target_size_mask)

        # aspect_ratio = img.size[1]/img.size[0]
        # aspect_ratio_mask = mask.size[1] / mask.size[0]

        # # Define the target size
        # target_height = 512
        # target_width = int(target_height * aspect_ratio)
        # target_width_mask = int(target_height * aspect_ratio_mask)

        # target_size = (target_height, target_width)
        # target_size_mask = (target_height,target_width_mask)

        # # Resize the image
        # img = img.resize(target_size)
        # mask = mask.resize(target_size_mask)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']

        if self.transform is None:
            img = Image.fromarray(img)

        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)
        # mask = Image.fromarray(np.array([self.fix_dict[x] for x in mask]))
        mask = torch.from_numpy(mask).long()

        if self.patches:
            img, mask = self.tiles(img, mask)

        return img, mask

    def tiles(self, img, mask):

        # img = img.transpose(2,0,1)
        # change n_s1 as you change the image shape dimensions
        n_sq = 1024
        # aspect_ratio = img.shape[1]/img.shape[0]
        aspect_ratio = 1024 / 800

        img_patches = img.unfold(1, n_sq, n_sq).unfold(2, (n_sq / aspect_ratio), (n_sq / aspect_ratio))
        img_patches = img_patches.contiguous().view(3, -1, (n_sq / aspect_ratio), (n_sq / aspect_ratio))
        img_patches = img_patches.permute(1, 0, 2, 3)

        mask_patches = mask.unfold(0, n_sq, n_sq).unfold(1, (n_sq / aspect_ratio), (n_sq / aspect_ratio))
        mask_patches = mask_patches.contiguous().view(-1, (n_sq / aspect_ratio), (n_sq / aspect_ratio))

        # img_patches = img.unfold(1, 512, 512).unfold(2, 768, 768)
        # img_patches  = img_patches.contiguous().view(3,-1, 512, 768)
        # img_patches = img_patches.permute(1,0,2,3)

        # mask_patches = mask.unfold(0, 512, 512).unfold(1, 768, 768)
        # mask_patches = mask_patches.contiguous().view(-1, 512, 768)

        # return img_patches, mask_patches

        return img_patches, mask_patches


if __name__ =='__main__':
    pass