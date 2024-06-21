from torch.utils.data import Dataset
from PIL import Image



class CarBrandDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.label_to_idx = {label: idx for idx, label in enumerate(dataframe['brand'].unique())}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['file_path']
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.iloc[idx]['brand']
        label_idx = self.label_to_idx[label]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label_idx
