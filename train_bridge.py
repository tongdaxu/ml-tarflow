import argparse
import pathlib
import builtins
import json
import torch
import torch.nn.functional as F
import torchvision as tv
import utils
import transformer_flow
from bridge_model import BridgeTokenizer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def add_input_noise(x, noise_type, noise_std):
    if noise_type == 'gaussian':
        eps = noise_std * torch.randn_like(x)
        return x + eps
    elif noise_type == 'uniform':
        x_int = (x + 1) * (255 / 2)
        x = (x_int + torch.rand_like(x_int)) / 256
        x = x * 2 - 1
        return x
    return x


def to_python_number(v):
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return v.item()
        return v.detach().float().mean().item()
    return float(v)


def main(args):
    dist = utils.Distributed()
    utils.set_random_seed(100 + dist.rank)

    if dist.rank == 0:
        args.logdir.mkdir(parents=True, exist_ok=True)

    data, num_classes = utils.get_data(args.dataset, args.img_size, args.data)

    def print(*args, **kwargs):
        if dist.rank == 0:
            builtins.print(*args, **kwargs, flush=True)

    writer = SummaryWriter(log_dir=str(args.logdir)) if dist.rank == 0 else None

    data_sampler = torch.utils.data.DistributedSampler(
        data, num_replicas=dist.world_size, rank=dist.rank, shuffle=True
    )
    data_loader = torch.utils.data.DataLoader(
        data,
        sampler=data_sampler,
        batch_size=args.batch_size // dist.world_size,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    # 1) load pretrained flow
    flow = transformer_flow.Model(
        in_channels=args.channel_size,
        img_size=args.img_size,
        patch_size=args.patch_size,
        channels=args.channels,
        num_blocks=args.blocks,
        layers_per_block=args.layers_per_block,
        nvp=args.nvp,
        num_classes=num_classes,
        use_checkpoint=False,
    ).cuda()

    ckpt = torch.load(args.flow_ckpt, map_location='cpu')
    flow.load_state_dict(ckpt)
    flow.eval()
    for p in flow.parameters():
        p.requires_grad_(False)

    flow_dim = args.channel_size * args.patch_size ** 2
    num_patches = (args.img_size // args.patch_size) ** 2

    # 2) trainable bridge
    bridge = BridgeTokenizer(
        flow_dim=flow_dim,
        num_patches=num_patches,
        core_dim=args.core_dim,
        depth=args.core_depth,
        head_dim=args.core_head_dim,
    ).cuda()

    if dist.distributed:
        bridge_ddp = torch.nn.parallel.DistributedDataParallel(
            bridge, device_ids=[dist.local_rank]
        )
    else:
        bridge_ddp = bridge

    optimizer = torch.optim.AdamW(
        bridge.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4
    )
    lr_schedule = utils.CosineLRSchedule(
        optimizer, len(data_loader), args.epochs * len(data_loader), 1e-6, args.lr
    )

    lpips_fn = None
    if args.use_lpips:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').cuda().eval()
        for p in lpips_fn.parameters():
            p.requires_grad_(False)

    for epoch in range(args.epochs):
        data_sampler.set_epoch(epoch)
        metrics = utils.Metrics()

        pbar = tqdm(data_loader, disable=not (dist.rank == 0))
        running_loss = running_lz = running_li = running_lp = 0.0

        for step, (x, y) in enumerate(pbar):
            x = x.cuda(non_blocking=True)
            x_in = add_input_noise(x, args.noise_type, args.noise_std)

            if num_classes:
                y = y.cuda(non_blocking=True)
            else:
                y = None

            with torch.no_grad():
                zf, _, _ = flow(x_in, y)

            zs, zf_hat = bridge_ddp(zf)

            zero = torch.zeros((), device=x.device)

            # latent loss
            if args.use_latent_loss:
                loss_z = F.mse_loss(zf_hat, zf)
            else:
                loss_z = zero

            # image / lpips 任一打开时 reverse
            need_recon = args.use_image_loss or args.use_lpips
            x_hat = None

            if need_recon:
                x_hat = flow.reverse(zf_hat, y, guidance=0)

                if args.use_image_loss:
                    loss_img = F.l1_loss(x_hat, x_in)
                else:
                    loss_img = zero

                if lpips_fn is not None:
                    loss_lp = lpips_fn(
                        x_hat.clamp(-1, 1),
                        x_in.clamp(-1, 1)
                    ).mean()
                else:
                    loss_lp = zero
            else:
                loss_img = zero
                loss_lp = zero

            loss = (
                args.lambda_z * loss_z
                + args.lambda_img * loss_img
                + args.lambda_lpips * loss_lp
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            current_lr = lr_schedule.step()

            metrics.update({
                'loss': loss.detach(),
                'loss/z': loss_z.detach(),
                'loss/img': loss_img.detach(),
                'loss/lpips': loss_lp.detach(),
                'lr': torch.tensor(current_lr, device=x.device),
                'zs/std': zs.detach().std(),
                'zs/abs_mean': zs.detach().abs().mean(),
            })

            running_loss += loss.item()
            running_lz += loss_z.item()
            running_li += loss_img.item()
            running_lp += loss_lp.item()

            if dist.rank == 0:
                pbar.set_postfix(
                    loss=f"{running_loss/(step+1):.4f}",
                    lz=f"{running_lz/(step+1):.4f}",
                    li=f"{running_li/(step+1):.4f}",
                    lp=f"{running_lp/(step+1):.4f}",
                    lr=f"{current_lr:.2e}",
                )

        metrics_dict = metrics.compute(dist)

        if dist.rank == 0:
            metrics_dict = {k: to_python_number(v) for k, v in metrics_dict.items()}
            metrics_dict['epoch'] = epoch + 1

            print(f"[Epoch {epoch+1}] {metrics_dict}")

            # 1) save structured metrics
            with open(args.logdir / 'metrics.jsonl', 'a') as f:
                f.write(json.dumps(metrics_dict) + '\n')

            # 2) TensorBoard
            if writer is not None:
                for k, v in metrics_dict.items():
                    if k != 'epoch':
                        writer.add_scalar(k, v, epoch + 1)
                writer.flush()

            # 3) save checkpoint
            bridge_to_save = bridge_ddp.module if hasattr(bridge_ddp, 'module') else bridge_ddp
            save_obj = {
                'epoch': epoch + 1,
                'model': bridge_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': vars(args),
                'metrics': metrics_dict,
            }
            torch.save(save_obj, args.logdir / 'bridge_latest.pth')
            torch.save(save_obj, args.logdir / f'bridge_epoch_{epoch+1:03d}.pth')

            # 4) save image
            if x_hat is not None:
                vis = torch.cat([x_in[:8], x_hat[:8]], dim=0)
                tv.utils.save_image(
                    vis,
                    args.logdir / f'bridge_recon_{epoch+1:03d}.png',
                    normalize=True,
                    value_range=(-1, 1),
                    nrow=8,
                )

        dist.barrier()

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data', type=pathlib.Path)
    parser.add_argument('--logdir', default='runs/bridge', type=pathlib.Path)
    parser.add_argument('--dataset', default='imagenet', choices=['imagenet', 'imagenet64', 'afhq'])
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--channel_size', default=3, type=int)

    parser.add_argument('--patch_size', default=8, type=int)
    parser.add_argument('--channels', default=768, type=int)
    parser.add_argument('--blocks', default=8, type=int)
    parser.add_argument('--layers_per_block', default=8, type=int)
    parser.add_argument('--noise_std', default=0.05, type=float)
    parser.add_argument('--noise_type', default='gaussian', choices=['gaussian', 'uniform'])
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--nvp', default=True, action=argparse.BooleanOptionalAction)

    parser.add_argument('--flow_ckpt', type=str, required=True)
    parser.add_argument('--core_dim', type=int, default=96)
    parser.add_argument('--core_depth', type=int, default=4)
    parser.add_argument('--core_head_dim', type=int, default=32)

    parser.add_argument('--lambda_z', type=float, default=1.0)
    parser.add_argument('--lambda_img', type=float, default=1.0)
    parser.add_argument('--lambda_lpips', type=float, default=0.1)
    parser.add_argument('--use_lpips', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--use_image_loss', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--use_latent_loss', action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    main(args)