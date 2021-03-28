import torch
import pickle

from torch.utils.data import Dataset
from typing import Any, Tuple, Dict, List
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import h5py
import torchvision.transforms as transforms


transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class MyDataset(Dataset):

    @staticmethod
    def create_images_h5py(directory_path):
        if not os.path.exists('data1'):
            os.mkdir('data1')
        path_images = f'data/images_old.h5'
        if os.path.exists(path_images):
            return path_images
        with h5py.File(path_images, "w-") as archive:
            for filename in tqdm(os.listdir(directory_path)):
                image_path = os.path.join(directory_path, filename)
                pil_img = Image.open(image_path).convert("RGB")
                img = np.array(pil_img.resize([64, 64]))
                img = img.transpose((2, 0, 1)).astype(np.float32)
                img = transform(torch.from_numpy(img))
                archive.create_dataset(image_path, data=img)
        return path_images

    @staticmethod
    def create_att_df(annotation_file):
        path_pkl = 'data/df_att.pkl'
        if os.path.exists(path_pkl):
            return pd.read_pickle(path_pkl)
        df = pd.read_csv(annotation_file, sep='  | ', engine='python', skiprows=[0])
        pickle.dump(df, open(path_pkl, 'wb'))
        return df

    def __init__(self, img_dir) -> None:
        self.paths = {
            'img_dir': img_dir,
            'images_h5py': MyDataset.create_images_h5py(img_dir),
            # 'annotation_file': annotation_file
        }

    def __getitem__(self, index: int) -> Tuple:
        image_path = f'{self.paths["img_dir"]}/{index + 1:06d}.jpg'
        return (transform(torch.from_numpy(h5py.File(self.paths['images_h5py'], "r")[image_path][:]))).tanh_()

    def __len__(self) -> int:
        """
        :return: the length of the dataset (number of sample).
        """
        # length = len(os.listdir(self.paths['img_dir']))
        return 202599
