from torch.utils.data import Dataset, DataLoader
from uilts.parse_cfg import parse_json
import os, torch
import random, cv2
import numpy as np
from augmentation.pipelines.compose import Compose


class ImageNetHard10DataSet(Dataset):
    def __init__(self, data_path, json_path, mode):

        config = parse_json(json_path)
        if mode == "train":
            image_path = os.path.join(data_path, 'train')
            self.train_pipeline = Compose(config['train'])
        else:
            image_path = os.path.join(data_path, 'val')
            self.train_pipeline = Compose(config['test'])

        self.image_datas = self._getData(image_path)

        # 随机打乱
        random.shuffle(self.image_datas)

    def __getitem__(self, item):
        img_path, label = self.image_datas[item]
        data = dict()

        x = cv2.imread(img_path)
        data["type"] = "classification"
        data["image"] = x
        augment_result = self.train_pipeline(data)

        img = augment_result["image"].astype(np.float32)
        img = torch.tensor(np.transpose(img, (2, 0, 1)), dtype=torch.float32)
        label = torch.tensor(int(label))
        return img, label

    def __len__(self):
        return len(self.image_datas)

    def _getData(self, image_paths):
        image_class_paths = os.listdir(image_paths)
        image_class_paths = sorted(image_class_paths)
        image_datas = []
        for i, image_class_path in enumerate(image_class_paths):
            image_path = os.path.join(image_paths, image_class_path)
            images = os.listdir(image_path)
            for image in images:
                if image.split('.')[1]=="JPEG":
                    image_data = [os.path.join(image_path, image), i]
                    image_datas.append(image_data)
        return image_datas


def denormalize(x_hat, mean=[0.2826372, 0.2826372, 0.2826372], std=[0.30690703, 0.30690703, 0.30690703]):

    mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1).cuda()
    std = torch.tensor(std).unsqueeze(1).unsqueeze(1).cuda()
    x = x_hat * std + mean
    return (x*255).cpu()


class PSC_Top_DataSet(Dataset):
    def __init__(self, data_path, json_path, mode, imshow=False):

        config = parse_json(json_path)
        if mode == "train":
            image_path = os.path.join(data_path, 'train')
            self.train_pipeline = Compose(config['train'])
        else:
            image_path = os.path.join(data_path, 'val')
            self.train_pipeline = Compose(config['test'])

        self.image_datas = self._getData(image_path)

        self.imshow = imshow
        # 随机打乱
        random.shuffle(self.image_datas)

    def __getitem__(self, item):
        img_path, label = self.image_datas[item]
        data = dict()

        x = cv2.imread(img_path)
        data["type"] = "classification"
        data["image"] = x
        augment_result = self.train_pipeline(data)

        img = augment_result["image"].astype(np.float32)
        img = torch.tensor(np.transpose(img, (2, 0, 1)), dtype=torch.float32)
        label = torch.tensor(int(label))

        if self.imshow:
            return img, label, img_path
        else:
            return img, label

    def __len__(self):
        return len(self.image_datas)

    def _getData(self, image_paths):
        image_class_paths = os.listdir(image_paths)
        image_class_paths = sorted(image_class_paths)
        image_datas = []
        for i, image_class_path in enumerate(image_class_paths):
            image_path = os.path.join(image_paths, image_class_path)
            images = os.listdir(image_path)
            for image in images:
                if image.split('.')[1] == "png":
                    image_data = [os.path.join(image_path, image), i]
                    image_datas.append(image_data)
        return image_datas


if __name__=="__main__":
    # json_path = "configs/1.json"
    # db = PSC_Top_DataSet(cfg.data_path, json_path, mode="val")
    #
    # loader = DataLoader(db, batch_size=1, shuffle=True)
    #
    # for x, y in loader:
    #     x = np.transpose(x, (0, 2, 3, 1))
    #     for i, image, in enumerate(x):
    #         image = denormalize(image, **{"mean": [123.799, 116.184, 100.685], "std": [66.172, 64.274, 66.643]})
    #         image = image.numpy() * 255
    #         image = image.astype(np.uint8)
    #         cv2.imshow('image', image)
    #
    #         print('label:', y[i])
    #         cv2.waitKey()
    #
    #
    pass
