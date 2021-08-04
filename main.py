import torch
from dataset.dataset import MetalNutDataset
import os
from torch import nn, optim
from torch.utils.data import DataLoader
from catalyst import dl, utils, metrics
from catalyst.data import ToTensor
from model.VAE import ResNetVAE
from torchvision import models, transforms
from utils.utils import cov, mahalanobis
from sklearn.metrics import roc_curve, roc_auc_score
import random
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
import numpy as np
from torch.nn import functional as F
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Anonymize analysis')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='device')
    parser.add_argument('--seed', type=int, default=100,
                        help='seed for reproducability')
    parser.add_argument('--data_path', type=str, default='./..',
                        help='path to the root directory containing data')
    parser.add_argument('--batchsize', type=int, default=32,
                        help='batchsize for dataloader')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--logs', type=str, default='./logs',
                        help='log directory')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cuda:
        if torch.cuda.is_available:
            device='cuda'
        else:
            print("Cuda not available, using cpu")
            device='cpu'
    else:
        device='cpu'
    utils.set_global_seed(args.seed)
    loaders, preprocess = generate_dataloaders(args.data_path, args.batchsize)
    model = models.resnet18(pretrained=True)
    roc_auc_resnet = compute_rocauc(model, loaders, device)
    vae = ResNetVAE(latent_dim=1000, freeze=True)
    vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    runner = CustomRunner()
    # model training
    runner.train(
        model=vae,
        criterion=criterion,
        optimizer=optimizer,
        loaders={ 'train': loaders['train'],
                  'valid': loaders['valid'],},
        num_epochs=args.epochs,
        logdir=args.logs,
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
        load_best_on_end=True,
    )
    model = vae.resnet18
    roc_auc_someth = compute_rocauc(model, loaders, device)
    for param in vae.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(vae.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    runner = CustomRunner()
    # model training
    runner.train(
        model=vae,
        criterion=criterion,
        optimizer=optimizer,
        loaders={ 'train': loaders['train'],
                  'valid': loaders['valid'],},
        num_epochs=args.epochs,
        logdir=args.logs,
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
        load_best_on_end=True,
    )
    model = vae.resnet18
    roc_auc_something = compute_rocauc(model, loaders, device)

    print('%s ROCAUC: %.3f' % ('metal nut, resnet', roc_auc_resnet))
    print('%s ROCAUC: %.3f' % ('metal nut, resnet.requires_grad=False', roc_auc_someth))
    print('%s ROCAUC: %.3f' % ('metal nut, full trained', roc_auc_something))



def compute_rocauc(model, loaders, device):
    """compute accuracy metrics for model
    Args:
        - model(torch.nn.Module): torch model
        - loaders(dict(torch.utils.data.DataLoader)): dataloaders
        - device(str): device to use
    Return:
        - Receiver Operating Characteristic/Area Under the Curve(float)
    """
    model = model.eval()
    model.to(device)

    def generate_embs(model, loader, device):
        embs_list = []
        labels_list = []
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                embs = model(imgs)
                embs_list.append(embs)
                labels_list.append(labels)
        embs = torch.cat(embs_list, 0)
        labels = torch.cat(labels_list, 0)
        return embs, labels

    embs_train, labels = generate_embs(model, loaders['train'], device)
    embs_valid, labels = generate_embs(model, loaders['valid'], device)
    embs_test, labels = generate_embs(model, loaders['test'], device)

    embs_train = torch.cat([embs_train, embs_valid], 0)
    mean = torch.mean(embs_train, dim=0).cpu().detach().numpy()
    covariance = LedoitWolf().fit(embs_train.cpu().detach().numpy()).covariance_
    embs_test = embs_test.cpu().detach().numpy()
    scores = []
    for emb in embs_test:
        scores.append(float(mahalanobis(emb, mean, np.linalg.inv(covariance))))
    gt_list = labels.cpu().detach().numpy()
    fpr, tpr, _ = roc_curve(gt_list, scores)
    roc_auc = roc_auc_score(gt_list, scores)
    return roc_auc

class CustomRunner(dl.Runner):
    def predict_batch(self, batch):
        return self.model(batch[0].to(self.device))

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveValueMetric(compute_on_call=False)
            for key in ["loss"]
        }

    def handle_batch(self, batch):
        # unpack the batch
        x, _ = batch
        # run model forward pass
        x_, _, _, _ = self.model(x)
        # compute the loss
        loss = F.mse_loss(x_, x)
        # log metrics
        self.batch_metrics.update(
            {"loss": loss}
        )
        for key in ["loss"]:
            self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)
        # run model backward pass
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def on_loader_end(self, runner):
        for key in ["loss"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)


def generate_dataloaders(path, batchsize):
    """generates list of images for training and testing

    Args:
        - path(str): root directory of the dataset
        - batchsize(int)
    Returns:
        - dictionary of dataloaders(train, val, test)
    """
    train_dir = os.path.join(path, 'train/good')
    imgs = [os.path.join(train_dir, img) for img in os.listdir(train_dir)]
    random.shuffle(imgs)
    train_imgs = imgs[: int(len(imgs) * .85)]
    val_imgs = imgs[int(len(imgs) * .85):]
    test_dir = os.path.join(path, 'test')
    test_dirs = os.listdir(test_dir)
    test_imgs = []
    test_labels = []

    for dirs in test_dirs:
        base_path = os.path.join(test_dir, dirs)
        imgs = os.listdir(base_path)
        if 'good' in dirs:
            test_labels.extend([0] * len(imgs))
            test_imgs.extend([os.path.join(base_path, img) for img in imgs])
        else:
            test_labels.extend([1] * len(imgs))
            test_imgs.extend([os.path.join(base_path, img) for img in imgs])
    batch_size = batchsize
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    loaders = {
        "train": DataLoader(
            MetalNutDataset(train_imgs, [0] * len(train_imgs), transform=preprocess),
            batch_size=batch_size
        ),
        "valid": DataLoader(
            MetalNutDataset(val_imgs, [0] * len(val_imgs), transform=preprocess),
            batch_size=batch_size
        ),
        "test": DataLoader(
            MetalNutDataset(test_imgs, test_labels, transform=preprocess),
            batch_size=batch_size,
            shuffle=False
        ),
    }
    return loaders, preprocess


if __name__ == '__main__':
    main()
