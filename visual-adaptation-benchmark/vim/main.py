# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import timm
import torch.backends.cudnn as cudnn
import json
from tqdm import tqdm
from torch.nn import functional as F
from pathlib import Path
from torch.optim import AdamW
import lily as lily
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.optim import create_optimizer_v2
from timm.utils import NativeScaler, get_state_dict, ModelEma
from timm.scheduler.cosine_lr import CosineLRScheduler
from datasets import build_dataset, get_data
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
from augment import new_data_aug_generator

from utils import AverageMeter
from contextlib import suppress

import models_mamba

import utils

@torch.no_grad()
def save(args, model, acc, ep):
    pass

def train(args, model, train_dl, test_dl, opt, scheduler, epoch):
    model.train()
    model = model.cuda()
    pbar = tqdm(range(epoch))
    for ep in pbar:
        model.train()
        model = model.cuda()
        for i, batch in enumerate(train_dl):
            x, y = batch[0].cuda(), batch[1].cuda()
            out = model(x)
            loss = F.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if scheduler is not None:
            scheduler.step(ep)
        if ep % 10 == 9:
            acc = test(model, test_dl)
            if acc > args.best_acc:
                args.best_acc = acc
            pbar.set_description('best_acc ' + str(args.best_acc) + '|' + str(acc))
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

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=10, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    
    # * Cosub params
    parser.add_argument('--cosub', action='store_true') 
    
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true') 
    
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data_set', default='cifar',
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # amp about
    parser.add_argument('--if_amp', action='store_true')
    parser.add_argument('--no_amp', action='store_false', dest='if_amp')
    parser.set_defaults(if_amp=True)

    # if continue with inf
    parser.add_argument('--if_continue_inf', action='store_true')
    parser.add_argument('--no_continue_inf', action='store_false', dest='if_continue_inf')
    parser.set_defaults(if_continue_inf=False)

    # if use nan to num
    parser.add_argument('--if_nan2num', action='store_true')
    parser.add_argument('--no_nan2num', action='store_false', dest='if_nan2num')
    parser.set_defaults(if_nan2num=False)

    # if use random token position
    parser.add_argument('--if_random_cls_token_position', action='store_true')
    parser.add_argument('--no_random_cls_token_position', action='store_false', dest='if_random_cls_token_position')
    parser.set_defaults(if_random_cls_token_position=False)    

    # if use random token rank
    parser.add_argument('--if_random_token_rank', action='store_true')
    parser.add_argument('--no_random_token_rank', action='store_false', dest='if_random_token_rank')
    parser.set_defaults(if_random_token_rank=False)

    # vit or vim
    parser.add_argument('--backbone', default='vim', type=str, choices=['vim','vit'])
    parser.add_argument('--vim_aug', action='store_true', help='Enable vim augmentation')
    parser.set_defaults(vim_aug=False)  
    parser.add_argument('--adapt_x', action='store_true', help='Enable x adaptation')
    parser.set_defaults(adapt_x=False)  
    parser.add_argument('--adapt_delta', action='store_true', help='Enable delta adaptation')
    parser.set_defaults(adapt_delta=False) 
    parser.add_argument('--adapt_out', action='store_true', help='Enable out adaptation')
    parser.set_defaults(adapt_out=False)  
    parser.add_argument('--out_low_rank', action='store_true', help='if adapt out, enabling low rank adaptation')
    parser.set_defaults(out_low_rank=False) 
    parser.add_argument('--adapt_in', action='store_true', help='Enable input adaptation')
    parser.set_defaults(adapt_in=False)  
    parser.add_argument('--dropout', action='store_true', help='Enable dropout')
    parser.set_defaults(dropout=False)
    parser.add_argument('--ne', default=1, type=int)
    parser.add_argument('--in_dim', default=4, type=int)
    parser.add_argument('--delta_dim', default=4, type=int)
    parser.add_argument('--s', default=1.0, type=float)
    return parser


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    print(f"seed is {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    print(f"lr {args.lr}")
    cudnn.benchmark = True
    
    data_loader_train, data_loader_val,nb_classes = get_data(args.data_set, normalize=False, vim_aug=args.vim_aug)

    if args.backbone == 'vim':
        print(f"Creating model: {args.model}")
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            img_size=args.input_size
        )

                        
        if args.finetune:
            if args.finetune.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.finetune, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.finetune, map_location='cpu')

            checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

            model.load_state_dict(checkpoint_model, strict=False)
            print("Finish loading checkpoints!")

    elif args.backbone == 'vit':
        print("creating model ViT-B/16_pretrained_on_ImageNet1K!")
        model = create_model('vit_base_patch16_224', checkpoint_path='../imagenet1k_ViT-B_16.npz', drop_path_rate=0.1)
        model.reset_classifier(nb_classes)

    if args.attn_only:
        for name_p,p in model.named_parameters():
            if '.attn.' in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            print('no position encoding')
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print('no patch embed')
            
    trainable = []
    paramcount = 0
    print(f"in_dim {args.in_dim}")
    print(f"delta dim {args.delta_dim}")
    lily.set_lily(model, args.in_dim, args.delta_dim, args.adapt_delta, adapt_in=args.adapt_in, ne=args.ne, dropout=args.dropout)
    for n, p in model.named_parameters():
        if ('adapter' in n or 'head' in n or 'router' in n):
            trainable.append(p)
            # print(n)
            p.requires_grad = True
            if 'head' in n:
                # print(n)
                pass
            if 'head' not in n:
                paramcount += p.numel()
        else:
            p.requires_grad = False

    # for n, p in model.named_parameters():
    #     if p.requires_grad == True:
    #         # print(f"name is {n}")
            
    model.to(device)
    print(paramcount)
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    # model_without_ddp = model

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    #     model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr

    # optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    optimizer = create_optimizer(args, model)
    # optimizer = create_optimizer_v2(model_without_ddp, optimizer_name='AdamW', learning_rate=5e-4,weight_decay=0.05)
    
    # amp about
    amp_autocast = suppress
    loss_scaler = "none"
    if args.if_amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    # lr_scheduler = CosineLRScheduler(optimizer, t_initial=100,
    #                                   warmup_t=5, lr_min=1e-5, warmup_lr_init=1e-6, decay_rate=0.1)
    lr_scheduler, _ = create_scheduler(args, optimizer)
            
    output_dir = Path(args.output_dir)
        
    print(f"Start training for 100 epochs on {args.data_set}")
    start_time = time.time()
    max_accuracy = 0.0
    args.best_acc = 0 
    model = train(args, model, data_loader_train, data_loader_val, optimizer, lr_scheduler, epoch=100)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if args.backbone == 'vim':
        with open('./vim.log', 'a') as f:
            f.write(f"dataset is {args.data_set} best acc is {args.best_acc}, s is {args.s}, lr is {args.lr}, params is {paramcount} weight decay is {args.weight_decay} dim is {args.in_dim} dropout is {args.dropout} \n")
    elif args.backbone == 'vit':
        with open('./vit.log', 'a') as f:
            f.write(f"dataset is {args.data_set} best acc is {args.best_acc}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)