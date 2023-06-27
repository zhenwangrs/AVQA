# coding=utf8
import logging
import os
import warnings

import munch
import numpy as np
import torch
import yaml
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_avqa_lstm import MavqaDataset_online
from models_lstm import LSTM_AVQA_Model

warnings.filterwarnings('ignore')


# torch.set_float32_matmul_precision('high')

def train():
    logger.info(config)

    model = LSTM_AVQA_Model(config).to(config.train.device)
    torch.save(model.state_dict(), f'./ckp/model_0.pth')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    scaler = GradScaler()

    optim = torch.optim.NAdam(model.parameters(), lr=config.train.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=config.train.epochs)

    start_epoch = config.train.start_epoch
    if config.train.start_epoch > 1:
        model.load_state_dict(torch.load(f'./ckp/model_{start_epoch - 1}.pth'))

    for i in range(start_epoch - 1):
        scheduler.step()

    if config.train.training_mode:
        train_dataset = MavqaDataset_online(config, model.tokenizer, model.image_processor, 'train')
        train_loader = DataLoader(train_dataset,
                                  batch_size=config.train.batch_size,
                                  shuffle=True,
                                  num_workers=6,
                                  prefetch_factor=2,
                                  persistent_workers=True,
                                  pin_memory=True,
                                  collate_fn=train_dataset.collate_fn,
                                  drop_last=False)

        for epoch in range(start_epoch, config.train.epochs + 1):
            epoch_loss = 0
            log_freq = len(train_loader) // config.train.log_freq
            for index, (text_input_ids, text_attention_mask, audio_feats, frame_feats,
                        labels, question_types, question_ids) in enumerate(
                tqdm(train_loader), start=1):
                text_input_ids = text_input_ids.to(config.train.device)
                text_attention_mask = text_attention_mask.to(config.train.device)
                audio_feats = audio_feats.to(config.train.device)
                frame_feats = frame_feats.to(config.train.device)
                labels = labels.to(config.train.device)
                with autocast():
                    loss, logits = model(text_input_ids, text_attention_mask, labels,
                                         audio_feats, frame_feats)
                scaler.scale(loss).backward()
                epoch_loss += loss.item()
                print(f'epoch: {epoch}, batch: {index}, loss: {loss.item()}, epoch_loss: {epoch_loss / index}')
                if index % config.train.batch_accum == 0:
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()
                if index % log_freq == 0:
                    logger.info(
                        f'epoch: {epoch}, batch: {index}, loss: {loss.item()}, epoch_loss: {epoch_loss / index}')
                torch.cuda.empty_cache()
            logger.info(f'epoch loss: {epoch_loss / index}')
            if epoch % config.train.save_freq == 0 and epoch >= config.train.start_save_epoch:
                torch.save(model.state_dict(), f'./ckp/model_{epoch}.pth')
            if epoch % config.train.test_freq == 0 and epoch >= config.train.start_test_epoch:
                test(model, config, split='test')

            scheduler.step()

    logger.info('testing ...')
    model.load_state_dict(torch.load(f'./ckp/model_{config.train.test_epoch}.pth'))
    test(model, config, split='test')


def test(model, config, split='test'):
    model.eval()

    test_dataset = MavqaDataset_online(config, model.tokenizer, model.image_processor, split)
    test_loader = DataLoader(test_dataset,
                             batch_size=config.train.batch_size,
                             shuffle=False,
                             num_workers=4,
                             prefetch_factor=2,
                             persistent_workers=False,
                             collate_fn=test_dataset.collate_fn,
                             drop_last=False)

    with torch.no_grad():
        answer_record = {}
        for index, (text_input_ids, text_attention_mask, audio_feats, frame_feats,
                    labels, question_types, question_ids) in enumerate(
            tqdm(test_loader), start=1):
            text_input_ids = text_input_ids.to(config.train.device)
            text_attention_mask = text_attention_mask.to(config.train.device)
            audio_feats = audio_feats.to(config.train.device)
            frame_feats = frame_feats.to(config.train.device)
            labels = labels.to(config.train.device)
            logits = model.predict(text_input_ids, text_attention_mask, labels,
                                   audio_feats, frame_feats)
            pred_score_list = torch.softmax(logits, dim=-1).cpu().tolist()
            for qid, pred_score, label, q_type in zip(question_ids, pred_score_list, labels, question_types):
                if qid not in answer_record:
                    answer_record[qid] = {
                        'pred_ans_score': pred_score,
                        'pred_ans_index': np.argmax(pred_score),
                        'correct_ans_index': label,
                        'question_type': q_type,
                    }
                    print(f'qid: {qid}, pred_ans_index: {np.argmax(pred_score)}, correct_ans_index: {label}')
                else:
                    raise ValueError(f'qid: {qid} has already in answer_record')

    # 计算准确率
    total_correct = 0
    for qid, record in answer_record.items():
        if record['correct_ans_index'] == np.argmax(record['pred_ans_score']):
            total_correct += 1
    logger.info(
        f'{split} total correct: {total_correct}, total: {len(answer_record)}, acc: {total_correct / len(answer_record)}')
    model.train()


if __name__ == '__main__':
    logging.basicConfig(filename='log.log', level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('t2i2p')
    logger.info('music avqa')
    chlr = logging.StreamHandler()  # 输出到控制台的handler
    chlr.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(chlr)

    config = yaml.load(open('config.yaml', 'r', encoding='utf-8'),
                       Loader=yaml.FullLoader)
    config = munch.munchify(config)

    os.environ['CUDA_VISIBLE_DEVICES'] = config.train.visible_gpu
    train()
