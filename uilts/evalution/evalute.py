import numpy as np
import torch


class AccScore():
    def __init__(self, model, loader, criteon, cfg):
        self.model = model
        self.loader = loader
        self.criteon = criteon
        self.device = cfg["device"]

    def __call__(self):
        self.model.eval()
        correct = 0
        total = len(self.loader.dataset)
        test_loss = 0.
        for x, y in self.loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                logits = self.model(x)
                pred = logits.argmax(dim=1)
            test_loss += self.criteon(logits, y).item()
            correct += torch.eq(pred, y).sum().float().item()

        return correct / total, test_loss / total


class F1Score():
    def __init__(self, model, loader, criteon, cfg):
        self.model = model
        self.loader = loader
        self.criteon = criteon
        self.num_classes = cfg["n_classes"]
        self.device = cfg["device"]

    def __call__(self):
        # 建立混淆矩阵
        confusion = np.zeros((self.num_classes, self.num_classes))
        # 计算 total_loss
        total_loss = 0.
        nLen = len(self.loader.dataset)
        # 前向推断
        with torch.no_grad():
            for i, (inputs, classes) in enumerate(self.loader):
                inputs = inputs.to(self.device)
                classes = classes.to(self.device)
                outputs = self.model(inputs)
                total_loss += self.criteon(outputs, classes).item()

                # 计算混淆矩阵
                preds = outputs.max(dim=1)[1].data.cpu().numpy().astype(int)
                classes = classes.data.cpu().numpy().astype(int)
                confusion += np.bincount(self.num_classes * classes + preds,
                                         minlength=self.num_classes**2).reshape((self.num_classes, self.num_classes))

        print(confusion)
        # To get the per-class accuracy: precision
        precision = np.diag(confusion) / (confusion.sum(0)+1e-4)

        recall = np.diag(confusion) / (confusion.sum(1)+1e-4)

        f1 = 2 * precision * recall / (precision + recall + 1e-4)

        return f1.mean(), total_loss/nLen

