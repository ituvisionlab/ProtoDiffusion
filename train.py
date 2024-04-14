import os
import torch
import argparse
import itertools
import numpy as np
from unet import Unet
from tqdm import tqdm
import torch.optim as optim
from diffusion import GaussianDiffusion
from utils import get_named_beta_schedule
from embedding import ConditionalEmbedding
from scheduler import GradualWarmupScheduler
from dataloaders import load_cifar10, load_stl10, load_tiny_imagenet
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import get_rank, init_process_group, destroy_process_group
from Models import ModifiedResNet
import random
import time 

def seed_everything(seed):   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train(params:argparse.Namespace):
    # set seed for reproducibility
    # seed_everything(params.seed)
    # set save dir    
    save_dir = f'results/{params.mode}/{params.dataset}/{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}'
    assert params.genbatch % (torch.cuda.device_count() * params.clsnum) == 0 , 'please re-set your genbatch!!!'
    # initialize settings
    init_process_group(backend="nccl")
    # get local rank for each process
    local_rank = get_rank()
    # set device
    device = torch.device("cuda", local_rank)
    # load data
    seed_everything(int(local_rank))
    
    if params.dataset == 'cifar10':
        dataloader, sampler = load_cifar10(params.batchsize, params.numworkers)
        class_number = 10
    elif params.dataset == 'stl10':
        dataloader, sampler = load_stl10(params.batchsize, params.numworkers,params.image_size)
        class_number = 10
    elif params.dataset == 'tiny-imagenet':
        dataloader, sampler = load_tiny_imagenet(params.batchsize, params.numworkers,params.image_size, params.tiny_imagenet_path)
        class_number = 200
        params.clsnum = 200
        params.genbatch = 800
    else:
        raise NotImplementedError

    # load checkpoint if resume training
    if params.path_resume != None:
        hyperparams_model = {}
        with open(os.path.join(params.path_resume, 'params.txt'), 'r') as f:
            for line in f.readlines():
                key, val = line.split(':')
                hyperparams_model[key] = val[:-1]
        net = Unet(
                in_ch = int(hyperparams_model["inch"]),
                mod_ch = int(hyperparams_model["modch"]),
                out_ch = int(hyperparams_model["outch"]),
                ch_mul = [ int(i)  for i in hyperparams_model["chmul"][2:-1].split(",")],
                num_res_blocks = int(hyperparams_model["numres"]),
                cdim = int(hyperparams_model["cdim"]),
                use_conv= True if hyperparams_model["useconv"] == ' True' else False,
                droprate = float(hyperparams_model["droprate"]),
                dtype=torch.float32
            ).to(device)
        
        print("Model is created")
       
        if os.path.exists(params.path_resume):
            save_dir = params.path_resume
            lastpath = params.path_resume + "/checkpoints"
            lastepc = torch.load(lastpath + "/last_epoch.pt")['last_epoch']
            # load checkpoints
            checkpoint = torch.load(os.path.join(lastpath, f'ckpt_{lastepc}_checkpoint.pt'), map_location='cpu')
            net.load_state_dict(checkpoint['net'])
            cemblayer.load_state_dict(checkpoint['cemblayer'])
            cemblayer.condEmbedding[0].weight.requires_grad = False
            print(f'load checkpoints from {lastpath}')
        else: 
            raise "path not exist"
    # new training
    else:
        # initialize models: select regular for classical diffusion training, proto for ProtoDiffusion
        if (params.mode == "regular" or params.mode == "proto_frozen" or params.mode == "proto_unfrozen"):
            net = Unet(
                        in_ch = params.inch,
                        mod_ch = params.modch,
                        out_ch = params.outch,
                        ch_mul = params.chmul,
                        num_res_blocks = params.numres,
                        cdim = params.cdim,
                        use_conv = params.useconv,
                        droprate = params.droprate,
                        dtype = params.dtype
                    )
        else:
            raise NotImplementedError
        
        cemblayer = ConditionalEmbedding(class_number, params.cdim, params.cdim)
        if (params.mode == "proto_frozen" or params.mode == "proto_unfrozen") and params.path_resume == None:
            # load prototypes from pretrained model
            model_proto = ModifiedResNet(params.cdim, s=2)
            model_proto = torch.load(params.path_proto)
            cemblayer.condEmbedding[0].weight = torch.nn.Parameter(torch.clone(model_proto.dce_loss.centers.detach()).T)
            # freeze or unfreeze prototypes
            if params.mode == "proto_frozen":       
                cemblayer.condEmbedding[0].weight.requires_grad = False
            elif params.mode == "proto_unfrozen":
                cemblayer.condEmbedding[0].weight.requires_grad = True
            print("Prototypes are loaded")
        
        cemblayer = cemblayer.to(device)

        lastepc = 0
        if local_rank == 0:
            os.makedirs(save_dir,exist_ok=True)
            os.makedirs(save_dir + '/samples',exist_ok=True)
            os.makedirs(save_dir + '/checkpoints',exist_ok=True)
            # write params to file
            with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
                for k, v in vars(params).items():
                    f.write(f'{k}: {v}\n')
    
    betas = get_named_beta_schedule(num_diffusion_timesteps = params.T)
    diffusion = GaussianDiffusion(
                    dtype = params.dtype,
                    model = net,
                    betas = betas,
                    w = params.w,
                    v = params.v,
                    device = device
                )   
    # DDP settings 
    diffusion.model = DDP(
                        diffusion.model,
                        device_ids = [local_rank],
                        output_device = local_rank
                    )
    cemblayer = DDP(
                    cemblayer,
                    device_ids = [local_rank],
                    output_device = local_rank
                )
    # optimizer settings
    optimizer = torch.optim.AdamW(
                    itertools.chain(
                        diffusion.model.parameters(),
                        cemblayer.parameters()
                    ),
                    lr = params.lr,
                    weight_decay = 1e-4
                )
    
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer = optimizer,
                            T_max = params.epoch,
                            eta_min = 0,
                            last_epoch = -1
                        )
    warmUpScheduler = GradualWarmupScheduler(
                            optimizer = optimizer,
                            multiplier = params.multiplier,
                            warm_epoch = params.epoch // 20,
                            after_scheduler = cosineScheduler,
                            last_epoch = lastepc
                        )
    
    if lastepc != 0:
        optimizer.load_state_dict(checkpoint['optimizer'])
        warmUpScheduler.load_state_dict(checkpoint['scheduler'])

    # training
    cnt = torch.cuda.device_count()
    # prototypes are freezed or continue training
    for epc in range(lastepc, params.epoch):
        # turn into train mode
        diffusion.model.train()
        cemblayer.train()
        sampler.set_epoch(epc)
        # batch iterations
        with tqdm(dataloader, dynamic_ncols=True, disable=(local_rank % cnt != 0)) as tqdmDataLoader:
            for img, lab in tqdmDataLoader:
                # write the epoch and iteration to log file
                f = open(os.path.join(save_dir, 'log.txt'), 'a')
                if local_rank == 0:
                    f.write(f'epoch: {epc + 1}, iteration: {tqdmDataLoader.n}\n')
                b = img.shape[0]
                optimizer.zero_grad()
                x_0 = img.to(device)
                lab = lab.to(device)
                cemb = cemblayer(lab)
                cemb[np.where(np.random.rand(b)<params.threshold)] = 0
                loss = diffusion.trainloss(x_0, cemb, file=f)
                loss.backward()
                optimizer.step()
                tqdmDataLoader.set_postfix(
                    ordered_dict={
                        "epoch": epc + 1,
                        "loss: ": loss.item(),
                        "batch per device: ":x_0.shape[0],
                        "img shape: ": x_0.shape[1:],
                        "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                    }
                )
                f.close()
        warmUpScheduler.step()
        # save checkpoint
        if (epc + 1) % params.interval == 0:
            checkpoint = {
                                'net':diffusion.model.module.state_dict(),
                                'cemblayer':cemblayer.module.state_dict(),
                                'optimizer':optimizer.state_dict(),
                                'scheduler':warmUpScheduler.state_dict()
                            }
            torch.save({'last_epoch':epc+1}, os.path.join(save_dir,'checkpoints/last_epoch.pt'))
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoints/ckpt_{epc+1}_checkpoint.pt'))
        torch.cuda.empty_cache()
    destroy_process_group()

def main():
    # several hyperparameters for model
    parser = argparse.ArgumentParser(description='test for diffusion model')

    parser.add_argument('--batchsize', type=int, default=256, help='batch size per device for training Unet model')
    parser.add_argument('--numworkers', type=int, default=4, help='num workers for training Unet model')
    parser.add_argument('--inch', type=int, default=3, help='input channels for Unet model')
    parser.add_argument('--modch', type=int, default=64, help='model channels for Unet model')
    parser.add_argument('--T', type=int, default=1000, help='timesteps for Unet model')
    parser.add_argument('--outch', type=int, default=3, help='output channels for Unet model')
    parser.add_argument('--chmul', type=list, default=[1,2,4,4], help='architecture parameters training Unet model')
    parser.add_argument('--numres', type=int, default=2, help='number of resblocks for each block in Unet model')
    parser.add_argument('--cdim', type=int, default=10, help='dimension of conditional embedding')
    parser.add_argument('--useconv', type=bool, default=True, help='whether use convlution in downsample')
    parser.add_argument('--droprate', type=float, default=0.1, help='dropout rate for model')
    parser.add_argument('--dtype', default=torch.float32)
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--w', type=float, default=1.8, help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--v', type=float, default=0.3, help='hyperparameters for the variance of posterior distribution')
    parser.add_argument('--epoch', type=int, default=1500, help='epochs for training')
    parser.add_argument('--multiplier', type=float, default=2.5, help='multiplier for warmup')
    parser.add_argument('--threshold', type=float, default=0.1, help='threshold for classifier-free guidance')
    parser.add_argument('--interval', type=int, default=20, help='checkpoint saving interval')
    parser.add_argument('--moddir', type=str, default='model', help='model addresses')
    parser.add_argument('--samdir', type=str, default='sample', help='sample addresses')
    parser.add_argument('--genbatch', type=int, default=80, help='batch size for sampling process')
    parser.add_argument('--clsnum', type=int, default=10, help='num of label classes')
    parser.add_argument('--ddim', type=lambda x:(str(x).lower() in ['true','1', 'yes']), default=False, help='whether to use ddim')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--image_size', default=-1, type=int, help='change image size')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset for training')
    parser.add_argument('--path_resume', default=None, type=str, help='Path of the checkpoint to resume training')
    parser.add_argument("--mode", default="regular", type=str, help="regular/proto_frozen/proto_unfrozen")
    parser.add_argument("--path_proto", default=None, type=str, help="Path of the checkpoint to prototypes")
    parser.add_argument("--tiny_imagenet_path", default=None, type=str, help="Path of the tiny imagenet dataset")
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
