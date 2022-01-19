import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer

import os
import numpy as np

import sys; sys.path.append('.')
from shared.data import get_dataloader
from shared.args import ARGS
from shared.constants import CUDA

import seq2seq.model as seq2seq_model
import seq2seq.utils as seq2seq_utils

import tagging.model as tagging_model
import tagging.utils as tagging_utils

import model as joint_model
import utils as joint_utils

"""
python joint/demo_stream.py --checkpoint model.ckpt --working_dir TEST --test bias_data/real_world_samples/speeches        --extra_features_top --pre_enrich --activation_hidden --tagging_pretrain_epochs 3        --pretrain_epochs 4 --learning_rate 0.0003 --epochs 20 --hidden_size 512 --train_batch_size 24        --test_batch_size 16 --bert_full_embeddings --debias_weight 1.3 --freeze_tagger True --token_softmax        --pointer_generator --coverage --inference_output TEST/TEST

"""


# # # # # # # # ## # # # ## # # DATA # # # # # # # # ## # # # ## # #
tokenizer = BertTokenizer.from_pretrained(ARGS.bert_model, cache_dir=ARGS.working_dir + '/cache')
tok2id = tokenizer.vocab
tok2id['<del>'] = len(tok2id)
id2tok = {x: tok for (tok, x) in tok2id.items()}


# # # # # # # # ## # # # ## # # MODEL # # # # # # # # ## # # # ## # #

if ARGS.pointer_generator:
    debias_model = seq2seq_model.PointerSeq2Seq(
        vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
        emb_dim=768, dropout=0.2, tok2id=tok2id) # 768 = bert hidden size
else:
    debias_model = seq2seq_model.Seq2Seq(
        vocab_size=len(tok2id), hidden_size=ARGS.hidden_size,
        emb_dim=768, dropout=0.2, tok2id=tok2id)


if ARGS.extra_features_top:
    tagging_model = tagging_model.BertForMultitaskWithFeaturesOnTop.from_pretrained(
            ARGS.bert_model,
            cls_num_labels=ARGS.num_categories,
            tok_num_labels=ARGS.num_tok_labels,
            cache_dir=ARGS.working_dir + '/cache',
            tok2id=tok2id)
elif ARGS.extra_features_bottom:
    tagging_model = tagging_model.BertForMultitaskWithFeaturesOnBottom.from_pretrained(
            ARGS.bert_model,
            cls_num_labels=ARGS.num_categories,
            tok_num_labels=ARGS.num_tok_labels,
            cache_dir=ARGS.working_dir + '/cache',
            tok2id=tok2id)
else:
    tagging_model = tagging_model.BertForMultitask.from_pretrained(
        ARGS.bert_model,
        cls_num_labels=ARGS.num_categories,
        tok_num_labels=ARGS.num_tok_labels,
        cache_dir=ARGS.working_dir + '/cache')


if ARGS.tagger_checkpoint:
    print('LOADING TAGGER FROM ' + ARGS.tagger_checkpoint)
    tagging_model.load_state_dict(torch.load(ARGS.tagger_checkpoint))
    print('DONE.')
if ARGS.debias_checkpoint:
    print('LOADING DEBIASER FROM ' + ARGS.debias_checkpoint)
    debias_model.load_state_dict(torch.load(ARGS.debias_checkpoint))
    print('DONE.')


joint_model = joint_model.JointModel(
    debias_model=debias_model, tagging_model=tagging_model)

if CUDA:
    joint_model = joint_model.cuda()

if ARGS.checkpoint is not None and os.path.exists(ARGS.checkpoint):
    print('LOADING FROM ' + ARGS.checkpoint)
    # TODO(rpryzant): is there a way to do this more elegantly?
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices
    if CUDA:
        joint_model.load_state_dict(torch.load(ARGS.checkpoint))
        joint_model = joint_model.cuda()
    else:
        joint_model.load_state_dict(torch.load(ARGS.checkpoint, map_location='cpu'))
    print('...DONE')


# # # # # # # # # # # # EVAL # # # # # # # # # # # # # #
def predict(s):
    pre_toks = tokenizer.tokenize(s)
    max_seq_len = 128

    with open('tmp', 'w') as f:
        f.write('\t'.join([
            'na',
            ' '.join(pre_toks),
            ' '.join(pre_toks),
            'na',
            'na'
        ]))

    dl, _ = get_dataloader('tmp', tok2id, 1)
    for batch in dl:
        (
            pre_id, pre_mask, pre_len,
            post_in_id, post_out_id,
            pre_tok_label_id, _,
            rel_ids, pos_ids, categories
        ) = batch

        post_start_id = tok2id['行']
        max_len = min(max_seq_len, pre_len[0].detach().cpu().numpy() + 10)

        with torch.no_grad():
            predicted_toks, predicted_probs = joint_model.inference_forward(
                pre_id, post_start_id, pre_mask, pre_len, max_len, pre_tok_label_id,
                rel_ids=rel_ids, pos_ids=pos_ids, categories=categories,
                beam_width=1)

        # print(predicted_toks); quit()
        pred_seq = [id2tok[x] for x in predicted_toks[0][1:]]
        if '止' in pred_seq:
            pred_seq = pred_seq[:pred_seq.index('止')]
        pred_seq = ' '.join(pred_seq).replace('[PAD]', '').strip()
        pred_seq = pred_seq.replace(' ##', '')
        return pred_seq


import time
while True:
    s = input("Enter a sentence:")
    start = time.time()
    print(predict(s))
    # print("Took %.2f seconds" % (time.time() - start))
quit()
