import logging
import sys

import munch
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

sys.path.append('../../siammae_pretrain')
from dataset_musicavqa_clip import MusicavqaDataset
from models_siammae_clip import SiamMAE


def train(config):
    model = SiamMAE(config)
    if config.train.start_epoch > 1:
        model.load_state_dict(torch.load(f'./ckp/model_{config.train.start_epoch-1}.pth'))
    model = model.cuda()

    dataset = MusicavqaDataset(config, model.image_processor, model.clip_image_processor)
    pretrain_dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=6,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
        collate_fn=dataset.collate_fn
    )

    optim = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=config.train.epochs)
    scaler = GradScaler()
    for epoch in range(config.train.start_epoch, config.train.epochs + 1):
        epoch_loss = 0
        log_freq = len(pretrain_dataloader) // config.train.log_freq
        for index, (audio_feat, frame1_feats, frame2_feats, frame2_clip_feats) in enumerate(tqdm(pretrain_dataloader), start=1):
            audio_feat = audio_feat.cuda()
            frame1_feats = frame1_feats.cuda()
            frame2_feats = frame2_feats.cuda()
            frame2_clip_feats = frame2_clip_feats.cuda()
            with autocast():
                total_loss, loss_frame2_recon, loss_audio_recon, clip_loss_frame, clip_loss_audio = model(audio_feat, frame1_feats, frame2_feats, frame2_clip_feats)
            print(f'epoch: {epoch}, batch: {index}, total loss: {total_loss.item()}, '
                            f'frame2 recon loss: {loss_frame2_recon.item()}, '
                            f'audio recon loss: {loss_audio_recon.item()}, '
                            f'clip loss frame: {clip_loss_frame.item()}, '
                            f'clip loss audio: {clip_loss_audio.item()}')
            epoch_loss += total_loss.item()
            scaler.scale(total_loss).backward()
            if index % config.train.batch_accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
            if index % log_freq == 0:
                logger.info(f'epoch: {epoch}, batch: {index}, total loss: {total_loss.item()}, '
                            f'frame2 recon loss: {loss_frame2_recon.item()}, '
                            f'audio recon loss: {loss_audio_recon.item()}, '
                            f'clip loss frame: {clip_loss_frame.item()}, '
                            f'clip loss audio: {clip_loss_audio.item()}')
        scheduler.step()
        logger.info(f'epoch: {epoch}, epoch_loss: {epoch_loss / len(pretrain_dataloader)}')
        if epoch % config.train.save_freq == 0:
            torch.save(model.state_dict(), f'./ckp/model_{epoch}.pth')


if __name__ == '__main__':
    logging.basicConfig(filename='pretrain.log', level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('siammae')
    chlr = logging.StreamHandler()  # 输出到控制台的handler
    chlr.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(chlr)

    config = yaml.load(open('config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    config = munch.munchify(config)

    logger.info(config)
    train(config)
