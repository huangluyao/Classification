import argparse, json, random, tqdm
import os, shutil, time
from uilts.log import get_logger
from uilts.dataset import get_dataset
from uilts.models import get_model
from uilts.loss import get_loss
from uilts.evalution import get_evalution
import torch
import torch.nn as nn


def run(cfg, logger):

    # 1. print info about configuration information
    logger.info(f'Conf | use dataset {cfg["dataset"]}')
    logger.info(f'Conf | use batch_size {cfg["batch_size"]}')
    logger.info(f'Conf | use model_name {cfg["model_name"]}')
    logger.info(f'Conf | use augmentation_path {cfg["augmentation_path"]}')

    # 2. load dataset
    train_loader, val_loader = get_dataset(cfg)

    # 3. load_model
    model = get_model(cfg["model_name"], cfg)

    # 4. whether to use multi-gpu training
    gpu_ids = [int(i) for i in list(cfg["gpu_ids"])]
    logger.info(f'Conf | use GPU {gpu_ids}')
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(cfg["device"])

    # 5. optimizer and learning rate decay
    logger.info(f'Conf | use optimizer Adam, lr={cfg["lr"]}, weight_decay={cfg["weight_decay"]}')
    logger.info(f'Conf | use step_lr_scheduler every {cfg["lr_decay_steps"]} steps decay {cfg["lr_decay_gamma"]}')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["lr_decay_steps"], gamma=cfg["lr_decay_gamma"])

    # 6. loss function and evaluation
    logger.info(f'Conf | use loss function {cfg["loss"]}')
    criterion = get_loss(cfg).to(cfg['device'])
    evalution = get_evalution(model, val_loader, criterion, cfg)

    # 7. train and val
    logger.info(f'Conf | use epoch {cfg["epoch"]}')
    best = 0.
    for epoch in range(cfg["epoch"]):
        model.train()
        train_loss = 0
        nLen = len(train_loader)
        qbar = tqdm.tqdm(train_loader, total=nLen)
        for i, (img, label) in enumerate(qbar):
            # load data to gpu
            img = img.to(cfg["device"])
            label = label.to(cfg["device"])
            # forward
            out = model(img)
            # calculate loss
            loss = criterion(out, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # trian_loss
            train_loss += loss.item()
            qbar.set_description("epoch %d, loss=%f" % (epoch, loss))

        scheduler.step()
        # val
        val_score, val_loss = evalution()
        if best <= val_score:
            best = val_score
            torch.save(model.state_dict(), os.path.join(cfg['logdir'], 'best_train.pth'))

        logger.info(f'Iter | [{epoch + 1:3d}/{cfg["epoch"]}] valid loss={val_loss:.5f}')
        logger.info(f'Test | [{epoch + 1:3d}/{cfg["epoch"]}] Valid score={val_score:.5f}')

    logger.info(f'END | best MIou in Test is  {best:.5f}')


if __name__=="__main__":

    # configuration parameter
    parse = argparse.ArgumentParser(description="config")
    parse.add_argument("--config",
                       nargs="?",
                       type=str,
                       default="configs/PSCTop_MobileNetv3.json")

    args = parse.parse_args()
    with open(args.config, 'r') as fp:
        cfg = json.load(fp)

    # Training Record
    logdir = f'run/{cfg["dataset"]}/{cfg["model_name"]}/{time.strftime("%Y-%m-%d-%H-%M")}-{random.randint(1000,10000)}'
    os.makedirs(logdir)
    shutil.copy(args.config, logdir)

    # get logger
    logger = get_logger(logdir)

    logger.info(f'Conf | user logdir {logdir}')
    cfg['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cfg['logdir'] = logdir

    run(cfg, logger)
