import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer

import os
import numpy as np

import spacy

import sys; sys.path.append('.')
from shared.data import get_dataloader_from_str
from shared.args import ARGS
from shared.constants import CUDA

import seq2seq.model as seq2seq_model
import seq2seq.utils as seq2seq_utils

import tagging.model as tagging_model
import tagging.utils as tagging_utils

import joint.model as joint_model
import joint.utils as joint_utils

from flask import Flask, jsonify, request, render_template

"""
python api/app.py \
  --checkpoint model.ckpt \
  --working_dir TEST \
  --test bias_data/real_world_samples/speeches \
  --extra_features_top \
  --pre_enrich \
  --activation_hidden \
  --tagging_pretrain_epochs 3 \
  --pretrain_epochs 4 \
  --learning_rate 0.0003 \
  --epochs 20 \
  --hidden_size 512 \
  --train_batch_size 24 \
  --test_batch_size 16 \
  --bert_full_embeddings \
  --debias_weight 1.3 \
  --freeze_tagger True \
  --token_softmax \
  --pointer_generator

"""

# # # # # # # # # # SETTINGS # # # # # # # # # # # #

nlp = spacy.load("en_core_web_sm")


# # # # # # # # ## # # # ## # # DATA # # # # # # # # ## # # # ## # #
tokenizer = BertTokenizer.from_pretrained(ARGS.bert_model, cache_dir=ARGS.working_dir + '/cache')
tok2id = tokenizer.vocab
tok2id['<del>'] = len(tok2id)


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

# # # # # # # # # # # # EVAL MODE & UTIL METHODS # # # # # # # # # # # # # #
joint_model.eval()
def words_from_toks(toks):
    words = []
    word_indices = []
    for i, tok in enumerate(toks):
        if tok.startswith('##'):
            words[-1] += tok.replace('##', '')
            word_indices[-1].append(i)
        else:
            words.append(tok)
            word_indices.append([i])
    return words, word_indices


def get_pos_dep(toks):
    out_pos, out_dep = [], []
    words, word_indices = words_from_toks(toks)
    analysis = nlp(' '.join(words))

    if len(analysis) != len(words):
        return None, None

    for analysis_tok, idx in zip(analysis, word_indices):
        out_pos += [analysis_tok.pos_] * len(idx)
        out_dep += [analysis_tok.dep_] * len(idx)

    assert len(out_pos) == len(out_dep) == len(toks)

    return ' '.join(out_pos), ' '.join(out_dep)


def transform_input(headline):
  tokenized = tokenizer.tokenize(headline)
  pos, deps = get_pos_dep(tokenized)
  final = '\t'.join([
            'na',
            ' '.join(tokenized),
            ' '.join(tokenized),
            'na',
            'na'
        ])

  return final

def load_data(transformed_input):
  eval_dataloader, num_eval_examples = get_dataloader_from_str(
    transformed_input,
    tok2id, ARGS.test_batch_size)

  print(eval_dataloader)
  print(num_eval_examples, flush=True)
  return eval_dataloader

def predict(dataloader):
  hits, preds, golds, srcs = joint_utils.run_eval(
    joint_model, dataloader, tok2id, None,
    ARGS.max_seq_len, ARGS.beam_width)

  print(hits, flush=True)
  print(preds, flush=True)
  print(golds, flush=True)
  return preds

# # # # # # # # # Server # # # # # # # # # #

app = Flask(__name__)

@app.route('/', methods=['GET'])
def root_route():
  return jsonify({'msg' : 'Try POSTing to the /predict endpoint with a url and headline text'})

@app.route('/test', methods=['GET'])
def test_route():
  return render_template('public/test.html')

@app.route('/predict', methods=['POST'])
def predict_route():
  req_data = request.get_json()

  headline = req_data['headline']
  transformed_input = transform_input(headline)

  print("\ntransformed_input: \n")
  print(transformed_input)
  print("\n")

  dataloader = load_data(transformed_input)
  prediction = predict(dataloader)

  words, _ = words_from_toks(prediction[0])
  return jsonify({'unbiased': ' '.join(words) })

if __name__ == '__main__':
  app.run(debug=True,host='0.0.0.0', port=5000)
  print("INFO: Server started")
