# coding=utf8
import json
import logging
import os
import warnings

import munch
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from dataset_mavqa_finetune_online import MavqaDataset_finetune_online
from dataset_mavqa_finetune_online_test import MavqaDataset_finetune_online_test
from models_avqa import FusVQAModel_online

warnings.filterwarnings("ignore")


# torch.set_float32_matmul_precision('high')

def train():
    logger.info(config)

    model = FusVQAModel_online(config).cuda()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    scaler = GradScaler()

    vit_params = list(model.vit.parameters())
    audio_mae_params = list(model.audio_mae.parameters())
    vilt_params = list(model.vilt.parameters())
    ans_mlp_params = list(model.ans_linear.parameters())

    #for param in (vit_params + audio_mae_params):
        #param.requires_grad = False

    params = [
        {'params': vit_params + audio_mae_params, 'lr': config.train.lr},
        {'params': vilt_params + ans_mlp_params, 'lr': config.train.lr},
    ]

    optim = torch.optim.Adam(params)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=config.train.epochs)

    start_epoch = config.train.start_epoch
    if config.train.start_epoch > 1:
        model.load_state_dict(torch.load(f'./ckp_online/model_16_same_lr.pth'), strict=False)

    for i in range(start_epoch - 1):
        scheduler.step()

    if config.train.epochs > 0:
        train_dataset = MavqaDataset_finetune_online(config, model.tokenizer, model.image_processor, 'train')
        train_loader = DataLoader(train_dataset,
                                  batch_size=config.train.batch_size,
                                  shuffle=True,
                                  num_workers=4,
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
                text_input_ids = text_input_ids.cuda()
                text_attention_mask = text_attention_mask.cuda()
                audio_feats = audio_feats.cuda()
                frame_feats = frame_feats.cuda()
                labels = labels.cuda()
                with autocast():
                    loss, logits = model(text_input_ids, text_attention_mask, labels,
                                         audio_feats, frame_feats)
                scaler.scale(loss).backward()
                print(f'epoch: {epoch}, loss: {loss.item()}, epoch_loss: {epoch_loss / index}')
                epoch_loss += loss.item()
                if index % config.train.batch_accum == 0:
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()
                if index % log_freq == 0:
                    logger.info(f'epoch: {epoch}, loss: {loss.item()}, epoch_loss: {epoch_loss / index}')
                torch.cuda.empty_cache()
            logger.info(f'epoch loss: {epoch_loss / index}')
            if epoch % config.train.save_freq == 0:
                torch.save(model.state_dict(), f'./ckp_online/model_{epoch}.pth')
            if epoch % config.train.test_freq == 0:
                test(model, config, split='test')
            scheduler.step()

    logger.info('testing ...')
    # model.load_state_dict(torch.load(f'./ckp_online/model_{config.train.epochs}.pth'))
    model.load_state_dict(torch.load(f'./ckp_online/model_16_same_lr.pth'), strict=False)
    # test(model, config, split='val')
    test_all(model, config, split='test')


def test(model, config, split='test'):
    model.eval()

    test_dataset = MavqaDataset_finetune_online(config, model.tokenizer, model.image_processor, split)
    test_loader = DataLoader(test_dataset,
                             batch_size=30,
                             shuffle=False,
                             num_workers=4,
                             prefetch_factor=2,
                             persistent_workers=True,
                             collate_fn=test_dataset.collate_fn,
                             drop_last=False)

    with torch.no_grad():
        answer_record = {}
        for index, (text_input_ids, text_attention_mask, audio_feats, frame_feats,
                    labels, question_types, question_ids) in enumerate(
            tqdm(test_loader), start=1):
            text_input_ids = text_input_ids.cuda()
            text_attention_mask = text_attention_mask.cuda()
            audio_feats = audio_feats.cuda()
            frame_feats = frame_feats.cuda()
            labels = labels.cuda()
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
                    # raise ValueError(f'qid: {qid} has already in answer_record')
                    print(f'qid: {qid} has already in answer_record')

    # 计算准确率
    total_correct = 0
    for qid, record in answer_record.items():
        if record['correct_ans_index'] == np.argmax(record['pred_ans_score']):
            total_correct += 1
    logger.info(
        f'{split} total correct: {total_correct}, total: {len(answer_record)}, acc: {total_correct / len(answer_record)}')
    model.train()



def test_all(model, config, split='test'):
    model.eval()
    answer_record = {}
    test_dataset = MavqaDataset_finetune_online_test(config, model.tokenizer, model.image_processor, split)
    test_loader = DataLoader(test_dataset,
                             batch_size=50,
                             shuffle=False,
                             num_workers=4,
                             prefetch_factor=2,
                             persistent_workers=True,
                             collate_fn=test_dataset.collate_fn,
                             drop_last=False)
    for i in range(1, 60 // config.model.select_num + 1):
        test_dataset.start_frame = i
        with torch.no_grad():
            for index, (text_input_ids, text_attention_mask, audio_feats, frame_feats,
                        labels, question_types, question_ids) in enumerate(
                tqdm(test_loader), start=1):
                text_input_ids = text_input_ids.cuda()
                text_attention_mask = text_attention_mask.cuda()
                audio_feats = audio_feats.cuda()
                frame_feats = frame_feats.cuda()
                labels = labels.cuda()
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
                        # print(f'qid: {qid}, pred_ans_index: {np.argmax(pred_score)}, correct_ans_index: {label}')
                    else:
                        answer_record[qid]['pred_ans_score'] = \
                            [x + y for x, y in zip(answer_record[qid]['pred_ans_score'], pred_score)]
                        answer_record[qid]['pred_ans_index'] = np.argmax(answer_record[qid]['pred_ans_score'])

        # 计算准确率
        total_correct = 0
        for qid, record in answer_record.items():
            if record['correct_ans_index'] == np.argmax(record['pred_ans_score']):
                total_correct += 1
        logger.info(
            f'{split} total correct: {total_correct}, total: {len(answer_record)}, acc: {total_correct / len(answer_record)}')

    model.train()


if __name__ == '__main__':
    logging.basicConfig(filename='log_online.log', level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('t2i2p')
    logger.info('music avqa')
    chlr = logging.StreamHandler()  # 输出到控制台的handler
    chlr.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(chlr)

    config = yaml.load(open('config.yaml', 'r', encoding='utf-8'),
                       Loader=yaml.FullLoader)
    config = munch.munchify(config)
    train()
