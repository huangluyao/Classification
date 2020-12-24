from uilts.dataset.CustomDataset import *


def get_ImageNetHard10DataSet(augmentation_path, batch_size):
    data_path = "./database/imagenet_hard10"
    db_train = ImageNetHard10DataSet(data_path, augmentation_path, mode="train")
    db_test = ImageNetHard10DataSet(data_path, augmentation_path, mode="val")

    train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(db_test, batch_size=batch_size)
    return train_loader, val_loader


def get_PSC_Top_DataSet(augmentation_path, batch_size):
    data_path = "./database/psa_top_rgb_cls"
    db_train = PSC_Top_DataSet(data_path, augmentation_path, mode="train")
    db_test = PSC_Top_DataSet(data_path, augmentation_path, mode="val")

    train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(db_test, batch_size=batch_size)
    return train_loader, val_loader


def get_dataset(cfg):

    return {"PSC_Top_DataSet": get_PSC_Top_DataSet,
            "ImageNetHard10DataSet": get_ImageNetHard10DataSet,
           }[cfg["dataset"]](cfg["augmentation_path"], cfg["batch_size"])



