from PIL import Image
from torch.utils.data import Dataset

class MetalNutDataset(Dataset):
    def __init__(self, imgs, labels, masks=None, transform=None):
        """Customer dataset
        Args:
            - imgs(list(str)): list of paths containing label images
            - labels(list(int)): list of labels
            - masks(list(str)): list of paths of mask images
            - transform(transform): depend on preprocessing required
        """
        self.masks = masks
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
    
    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        if self.masks:
            if 'good' in self.imgs[idx]:
                return img, Image.open(self.imgs[idx].replace('test', 'ground_truth'))
            else:
                _, n, m = img.shape
                return img, Image.new('RGB', (n, m))
        return  img, label
    
    def __len__(self):
        return len(self.imgs)
