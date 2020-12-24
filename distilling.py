import argparse, json, time, os
import shutil, random ,tqdm
from uilts.log import get_logger
from uilts.models import get_model
from uilts.dataset import get_dataset
from uilts.loss import get_loss
from uilts.evalution import get_evalution
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F


def run(cfg, logger):
    # 1. print info about configuration information
    T, weight_stu = cfg["T"], cfg["weight_student"]
    logger.info(f'Conf | use batch_size {cfg["batch_size"]}')
    logger.info(f'Conf | use distilling parameter T= {T}')
    logger.info(f'Conf | use distilling parameter weight_student= {weight_stu}')

    # 2. load dataset
    logger.info(f'Conf | use dataset {cfg["dataset"]}')
    logger.info(f'Conf | use augmentation_path {cfg["augmentation_path"]}')
    train_loader, val_loader = get_dataset(cfg)

    # 3. load teacher model and student model
    logger.info(f'Conf | use teacher_model: {cfg["teacher_model"]}')
    teacher_model = get_model(cfg["teacher_model"], cfg)
    teacher_dict = torch.load(cfg["teacher_weights"])
    teacher_model.load_state_dict(teacher_dict)

    logger.info(f'Conf | use student_model: {cfg["student_model"]}')
    student_model = get_model(cfg["student_model"], cfg)
    student_model.load_state_dict(torch.load(cfg["student_weights"]))

    # 4. whether to use multi-gpu training
    gpu_ids = [int(i) for i in list(cfg["gpu_ids"])]
    logger.info(f'Conf | use GPU {gpu_ids}')
    if len(gpu_ids) > 1:
        teacher_model = nn.DataParallel(teacher_model, device_ids=gpu_ids)
        student_model = nn.DataParallel(student_model, device_ids=gpu_ids)

    student_model = student_model.to(cfg["device"])
    teacher_model = teacher_model.to(cfg["device"])

    # 5. optimizer and learning rate decay
    logger.info(f'Conf | use optimizer Adam, lr={cfg["lr"]}, weight_decay={cfg["weight_decay"]}')
    logger.info(f'Conf | use step_lr_scheduler every {cfg["lr_decay_steps"]} steps decay {cfg["lr_decay_gamma"]}')
    optimizer = torch.optim.Adam(student_model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["lr_decay_steps"], gamma=cfg["lr_decay_gamma"])

    # 6. loss function and evaluation
    loss_KD = nn.KLDivLoss().to(cfg["device"])
    logger.info(f'Conf | use loss function KLDivLoss')
    logger.info(f'Conf | use loss function {cfg["loss"]}')
    criterion = get_loss(cfg).to(cfg['device'])
    evalution = get_evalution(student_model, val_loader, criterion, cfg)

    # 7. train and val
    logger.info(f'Conf | use epoch {cfg["epoch"]}')
    best = 0.
    teacher_model.eval()
    for epoch in range(cfg["epoch"]):
        student_model.train()
        nLen = len(train_loader)
        qbar = tqdm.tqdm(train_loader, total=nLen)
        for i , (image, label) in enumerate(qbar):
            image = Variable(image.to(cfg["device"]))
            label = Variable(label.to(cfg["device"]))

            # froward
            y_teacher = teacher_model(image)
            y_student = student_model(image)

            loss_student = criterion(y_student, label)
            loss_teacher = loss_KD(F.log_softmax(y_teacher/T, dim=1),
                                   F.softmax(y_student/T, dim=1))

            loss = weight_stu*loss_student + (1-weight_stu)*T*T*loss_teacher
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # trian_loss
            qbar.set_description("epoch %d, loss=%f" % (epoch, loss))

        scheduler.step()
        # val
        val_score, val_loss = evalution()
        if best <= val_score:
            best = val_score
            torch.save(student_model.state_dict(), os.path.join(cfg['logdir'], 'best_train.pth'))

        logger.info(f'Iter | [{epoch + 1:3d}/{cfg["epoch"]}] valid loss={val_loss:.5f}')
        logger.info(f'Test | [{epoch + 1:3d}/{cfg["epoch"]}] Valid score={val_score:.5f}')

    logger.info(f'END | best MIou in Test is  {best:.5f}')


if __name__=="__main__":
    parse = argparse.ArgumentParser(description="config")
    parse.add_argument("--config",
                       nargs="?",
                       type=str,
                       default="configs/distilling.json",
                       help="distilling config")

    args = parse.parse_args()
    with open(args.config, 'r') as fp:
        cfg = json.load(fp)

    # Training Record
    logdir = f'run/{cfg["dataset"]}/Distilling/{time.strftime("%Y-%m-%d-%H-%M")}-{random.randint(1000,10000)}'
    os.makedirs(logdir)
    shutil.copy(args.config, logdir)

    # get logger
    logger = get_logger(logdir)

    logger.info(f'Conf | user logdir {logdir}')
    cfg['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cfg['logdir'] = logdir

    run(cfg, logger)