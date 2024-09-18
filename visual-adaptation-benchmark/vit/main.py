import torch
from torch import nn
from torch.optim import AdamW
from torch.nn import functional as F
from tqdm import tqdm
import timm
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
from vtab import *
from utils import set_seed, AverageMeter
import adaptformer
import lora
import lily
import wandb
os.environ["WANDB_MODE"] = "offline"
import os

def train(args, model, dl, opt, scheduler, epoch):
    model.train()
    model = model.cuda()
    pbar = tqdm(range(epoch))
    for ep in pbar:
        model.train()
        model = model.cuda()
        for i, batch in enumerate(dl):
            x, y = batch[0].cuda(), batch[1].cuda()
            out = model(x)
            loss = F.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if scheduler is not None:
            scheduler.step(ep)
        if ep % 10 == 9:
            acc = test(vit, test_dl)
            wandb.log({"accuracy": acc, "loss": loss})
            if acc > args.best_acc:
                args.best_acc = acc
            pbar.set_description('best_acc ' + str(args.best_acc))

    model = model.cpu()
    return model


@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = AverageMeter()
    model = model.cuda()
    for batch in tqdm(dl):
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        acc.update(out, y)
    return acc.result().item()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--ne', type=int, default=4)
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--method', type=str, default='adaptformer',
                        choices=['adaptformer', 'lora', 'lily'])
    parser.add_argument('--monoscale', action='store_true', default=False)
    parser.add_argument('--model_path', type=str, default='.')
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)
    args.best_acc = 0
    vit = create_model(args.model, checkpoint_path='./ViT-B_16.npz', drop_path_rate=0.1)
    train_dl, test_dl = get_data(args.dataset, normalize=False)

    if args.method == 'adaptformer':
        adaptformer.set_adapter(vit, dim=args.dim, s=args.scale, bit=args.bit)
        vit.reset_classifier(get_classes_num(args.dataset))
    elif args.method == 'lora':
        lora.set_adapter(vit, dim=args.dim, s=args.scale, bit=args.bit)
        vit.reset_classifier(get_classes_num(args.dataset))
    elif args.method == 'lily':
        if not args.monoscale:
            lily.set_lily_kv(vit, args.dim, args.scale, args.ne)
        else:
            lily.set_lily_kv_monoscale(vit, args.dim, args.scale, args.ne)
        vit.reset_classifier(get_classes_num(args.dataset))

    param = 0

    trainable = []
    for n, p in vit.named_parameters():
        if 'head' in n or 'lily' in n:
            trainable.append(p)
            if 'head' not in n:
                param += p.numel()
        else:
            p.requires_grad = False
    opt = AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    scheduler = CosineLRScheduler(opt, t_initial=100,
                                    warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6, decay_rate=0.1)
    
    print(f"param: {param}")
    print(f"scale is {args.scale}")
    print(f"monoscale is {args.monoscale}")
    
    wandb.init(project="lily", name=f"ne-{args.ne}-dim-{args.dim}-scale-{args.scale}-dataset-{args.dataset}-monoscale-{args.monoscale}")
    wandb.config.update(args)
    
    vit = train(args, vit, train_dl, opt, scheduler, epoch=100)

    print('best_acc:', args.best_acc)
    if not args.monoscale:
        with open('./lily.log', 'a') as f:
            f.write(f"dataset {args.dataset}, acc {args.best_acc}, scale {args.scale}, ne {args.ne}, params {param}, monoscale {args.monoscale}\n")
    else:
        with open('./lily_mono.log', 'a') as f:
            f.write(f"dataset {args.dataset}, acc {args.best_acc}, scale {args.scale}, ne {args.ne}, params {param}, monoscale {args.monoscale}\n")

