from tqdm import tqdm
import utils
import os
import random
import argparse
import numpy as np
import sys

from torch.utils import data
from datasets.LC8_FP_SEG_V4 import LC8FPSegDataset
from utils import ext_transforms as et
from metrics import StreamSegMetrics
from network.FPSFormer import FPSFormer
import torch.nn.functional as F
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
import pytorch_ssim
import pytorch_iou

def get_argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--trainset_path", type=str, 
        default='D:/2023_Files/Dataset/LC8FPS1K_Aug/Train',
        help="path to Dataset")
    parser.add_argument("--testset_path", type=str, 
        default='D:/2023_Files/Dataset/LC8FPS1K_Aug/Test',
        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='LC8FPS1K_Aug', help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=2,
                        help='num_classes')
    parser.add_argument("--phi", type=str, default='b1',
                       help='MIT version from b1-b5')
    
    parser.add_argument("--model", type=str, default='FPSFormer',
        help='model name:[FPSFormer]')
    parser.add_argument("--threshold", type=float, default=0.6,
                     help='threshold to predict foreground')
    
    # Train Options 
    parser.add_argument("--epochs", type=int, default=120,
                        help="epoch number (default: 120)")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10)
  
    parser.add_argument("--batch_size", type=int, default=2,
                        help='batch size ')
    parser.add_argument("--crop_size", type=int, default=1024)

    parser.add_argument("--n_cpu", type=int, default=2,
                        help="download datasets")
    
    parser.add_argument("--ckpt", type=str,
            default=None, help="restore from checkpoint")
    parser.add_argument("--loss_type", type=str, default='bsi', help="loss type")
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--val_interval", type=int, default=5,
                        help="epoch interval for eval (default: 100)")
    return parser


def get_dataset(opts):
    train_transform = et.ExtCompose([
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    val_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    train_dst = LC8FPSegDataset(is_train=True,voc_dir=opts.trainset_path, 
                                transform=train_transform)
    val_dst = LC8FPSegDataset(is_train=False,voc_dir=opts.testset_path,
                            transform=val_transform)
    return train_dst, val_dst


def validate(opts, model, loader, device, metrics):
    metrics.reset()
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            SOUTS=model(images)
              
            outputs=SOUTS[0]
            outputs=outputs.squeeze()
            outputs[outputs>=opts.threshold]=1
            outputs[outputs<opts.threshold]=0
            
            preds = outputs.detach().cpu().numpy()
            
            preds=preds.astype(int)
            targets = labels.cpu().numpy()
            if targets.shape[0]==1:
                targets=targets.squeeze()
           
            metrics.update(targets, preds)   
 
        score = metrics.get_results()
    return score


bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def bce_ssim_loss(pred,target):
	bce_out = bce_loss(pred,target)
	ssim_out = 1 - ssim_loss(pred,target)
	iou_out = iou_loss(pred,target)

	loss = bce_out + ssim_out + iou_out

	return loss

def bsi_DS6(d0, d1, d2, d3, d4, d5, labels_v):
    loss0 = bce_ssim_loss(d0,labels_v)
    loss1 = bce_ssim_loss(d1,labels_v)
    loss2 = bce_ssim_loss(d2,labels_v)
    loss3 = bce_ssim_loss(d3,labels_v)
    loss4 = bce_ssim_loss(d4,labels_v)
    loss5 = bce_ssim_loss(d5,labels_v)
   
    loss = loss0 + loss1 + loss2 + loss3/2 + loss4/4 + loss5/8
    return loss

def main():
    if not os.path.exists('CHKP'):
        utils.mkdir('CHKP')

    opts = get_argparser().parse_args()

    tb_writer = SummaryWriter()
    
    torch.cuda.empty_cache()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    train_dst, val_dst = get_dataset(opts)
    opts.total_itrs=opts.epochs * (len(train_dst) // opts.batch_size)
    print('opts:',opts)

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.n_cpu,
        drop_last=True)  
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.n_cpu)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))


    model = FPSFormer(n_channels=3, phi=opts.phi)
   
    metrics = StreamSegMetrics(opts.num_classes)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
 
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    def save_ckpt(path):
        torch.save({
            "epoch": epoch+1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }, path)
        print("Model saved as %s" % path)  
        
    cur_epoch=0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model=model.to(device)
        
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        cur_epoch = checkpoint["epoch"]   
        
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model=model.to(device)


    for epoch in range(cur_epoch,opts.epochs):
        model.train()
        cur_itrs=0
        data_loader = tqdm(train_loader, file=sys.stdout)
        running_loss = 0.0
        
        for (images, labels) in data_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            optimizer.zero_grad()

            labels=labels.unsqueeze(dim=1)
           
            Sout, Sfuse, S1, S2, S3, S4 = model(images)
            total_loss=bsi_DS6(Sout, Sfuse, S1, S2, S3, S4, labels)
            
            running_loss += total_loss.data.item()

            total_loss.backward()
            optimizer.step()

            data_loader.desc = "Epoch {}/{}, loss={:.4f}".format(epoch, opts.epochs, running_loss/cur_itrs)
            
            scheduler.step()
          
        if (epoch+1) % opts.val_interval == 0:
            save_ckpt('CHKP/latest_{}_{}_{}.pth'.format(opts.model, opts.phi, 
                                        opts.dataset))
            if (epoch+1) % 20 == 0 :
                save_ckpt('CHKP/{}_{}_{}_ep{}.pth'.format(opts.model, opts.phi, 
                                    opts.dataset, epoch+1))
        
        print("validation...")
        model.eval()
        val_score = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)
        
        print('val_score:',val_score)
    
        tags = ["train_loss", "learning_rate","Mean_Acc","Mean_IoU","Fire_IoU"]

        tb_writer.add_scalar(tags[0], (running_loss/cur_itrs), epoch)
        tb_writer.add_scalar(tags[1], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[2], val_score['Mean Acc'], epoch)
        tb_writer.add_scalar(tags[3], val_score['Mean IoU'], epoch)
        tb_writer.add_scalar(tags[4], val_score['Class IoU'][1], epoch)
    

if __name__ == '__main__':
    main()
