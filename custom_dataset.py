from torch_snippets import *
from PIL import Image
import glob
import os
from utils import preprocess_image

class OpenDataset(torch.utils.data.Dataset):
    w, h = 224, 224
    def __init__(self, df, root_dir, label2target):
        self.root_dir = root_dir
        self.df = df
        self.label2target = label2target
    
    def __getitem__(self, ix):
        img_path = os.path.join(self.root_dir, self.df.iloc[ix, 0])
        img = Image.open(img_path).convert("RGB")
        H, W, C = np.array(img).shape
        img = np.array(img.resize((self.w, self.h), resample=Image.BILINEAR)) / 255.
        bbox = (self.df.iloc[ix, 3:7].values) / [W, H, W, H]
        bbox = bbox * [self.w, self.h, self.w, self.h]
        bbox = np.expand_dims(bbox, axis=0)
        bbox = bbox.tolist()
        target = {}
        target['boxes'] = torch.Tensor(bbox).float()
        target['labels'] = torch.Tensor(self.label2target[self.df.iloc[ix, 0].split('/')[0]]).long()
        img = preprocess_image(img)
        return img, target
    
    def collate_fn(self, batch):
        return tuple(zip(*batch))
    
    def __len__(self):
        return len(self.df)