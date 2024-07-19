# -*- encoding: utf-8 -*-
'''
@File     : ddp_mner.py
@DateTime : 2020/08/31 00:06:36
@Author   : Swift
@Desc     : reset twitter2015 model
'''


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms
from transformers import BertTokenizer
from transformers import BertModel
from torchcrf import CRF
from PIL import Image
import os
import glob
import numpy as np
import time
import random,pdb
import argparse
from tqdm import tqdm
import warnings, time
from model.utils import *
from metric import evaluate_pred_file
from config import tag2idx, idx2tag, max_len, max_node, log_fre


import flask
from flask import Flask, jsonify, request
warnings.filterwarnings("ignore")
predict_file = "./output/twitter2017/{}/epoch_{}.txt"
device = torch.device("cpu")
token =  BertTokenizer.from_pretrained('bert-base-cased')

def seed_torch(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class MMNerDataset(Dataset):
    def __init__(self, sentence, image):
        self.sentence = sentence
        self.image  = image
        self.tokenizer = token

    def __len__(self):
        return 1

    def construct_inter_matrix(self, word_num, pic_num=max_node):
        mat = np.zeros((max_len, pic_num), dtype=np.float32)
        mat[:word_num, :pic_num] = 1.0
        return mat

    def __getitem__(self, idx):
        # with open(self.X_files[idx], "r", encoding="utf-8") as fr:
        #     s = fr.readline().split()
        s = self.sentence.split()
        l = ['O' for i in s]
      
        picpaths = [self.image, self.image, self.image,self.image]

        ntokens = ["[CLS]"]
        label_ids = [tag2idx["CLS"]]
        for word, label in zip(s, l):    # iterate every word
            tokens = self.tokenizer._tokenize(word)    # one word may be split into several tokens
            ntokens.extend(tokens)
            for i, _ in enumerate(tokens):
                label_ids.append(tag2idx[label] if i == 0 else tag2idx["X"])
        ntokens = ntokens[:max_len-1]
        ntokens.append("[SEP]")
        label_ids = label_ids[:max_len-1]
        label_ids.append(tag2idx["SEP"])

        matrix = self.construct_inter_matrix(len(label_ids), len(picpaths))

        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
        mask = [1] * len(input_ids)
        segment_ids = [0] * max_len

        pad_len = max_len - len(input_ids)
        rest_pad = [0] * pad_len    # pad to max_len
        input_ids.extend(rest_pad)
        mask.extend(rest_pad)
        label_ids.extend(rest_pad)

        # pad ntokens
        ntokens.extend(["pad"] * pad_len)

        return {
            "ntokens": ntokens,
            "input_ids": input_ids,
            "segment_ids": segment_ids,
            "mask": mask,
            "label_ids": label_ids,
            "picpaths": picpaths,
            "matrix": matrix
        }


def collate_fn(batch):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    label_ids = []

    b_ntokens = []
    b_matrix = []
    b_img = torch.zeros(len(batch)*max_node, 3, 224, 224)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    for idx, example in enumerate(batch):
        b_ntokens.append(example["ntokens"])
        input_ids.append(example["input_ids"])
        token_type_ids.append(example["segment_ids"])
        attention_mask.append(example["mask"])
        label_ids.append(example["label_ids"])
        b_matrix.append(example["matrix"])

        for i, picpath in enumerate(example["picpaths"]):
            try:
                b_img[idx*max_node+i] = preprocess(Image.open(picpath.stream).convert('RGB'))
            except:
                print("========={} error!===============".format(picpath))
                exit(1)
            

    return {
        "b_ntokens": b_ntokens,
        "x": {
            "input_ids": torch.tensor(input_ids).to(device),
            "token_type_ids": torch.tensor(token_type_ids).to(device),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.uint8).to(device)
        },
        "b_img": torch.tensor(b_img).to(device),
        "b_matrix": torch.tensor(b_matrix).to(device),
        "y": torch.tensor(label_ids).to(device)
    }


class MMNerModel(nn.Module):

    def __init__(self, d_model=512, d_hidden=256, n_heads=8, dropout=0.4, layer=6, tag2idx=tag2idx):
        super(MMNerModel, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.resnet = models.resnet152(pretrained=True)
        self.crf = CRF(len(tag2idx), batch_first=True)
        # self.hidden2tag = nn.Linear(2*d_model, len(tag2idx))
        self.hidden2tag = nn.Linear(768+512, len(tag2idx))

        objcnndim = 2048
        fc_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(
            in_features=fc_feats, out_features=objcnndim, bias=True)

        self.layer = layer
        self.dp = dropout
        self.d_model = d_model
        self.hid = d_hidden

        self.trans_txt = nn.Linear(768, d_model)
        self.trans_obj = nn.Sequential(Linear(objcnndim, d_model), nn.ReLU(), nn.Dropout(dropout),
                                       Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout))

        # text
        self.mhatt_x = clone(MultiHeadedAttention(
            n_heads, d_model, dropout), layer)
        self.ffn_x = clone(PositionwiseFeedForward(d_model, d_hidden), layer)
        self.res4ffn_x = clone(SublayerConnectionv2(d_model, dropout), layer)
        self.res4mes_x = clone(SublayerConnectionv2(d_model, dropout), layer)

        # img
        self.mhatt_o = clone(MultiHeadedAttention(
            n_heads, d_model, dropout, v=0, output=0), layer)
        self.ffn_o = clone(PositionwiseFeedForward(d_model, d_hidden), layer)
        self.res4mes_o = clone(SublayerConnectionv2(d_model, dropout), layer)
        self.res4ffn_o = clone(SublayerConnectionv2(d_model, dropout), layer)

        self.mhatt_x2o = clone(Linear(d_model * 2, d_model), layer)
        self.mhatt_o2x = clone(Linear(d_model * 2, d_model), layer)
        self.xgate = clone(SublayerConnectionv2(d_model, dropout), layer)
        self.ogate = clone(SublayerConnectionv2(d_model, dropout), layer)

    def log_likelihood(self, x, b_img, inter_matrix, text_mask, tags):
        """
        inter_matrix: batch, max_len, 4
        text_mask: batch, max_len
        """
        batch_size, xn, objn = inter_matrix.size(0), inter_matrix.size(1), inter_matrix.size(2)
        inter_matrix = inter_matrix.unsqueeze(-1)
        matrix4obj = torch.transpose(inter_matrix, 1, 2)
        crf_mask = x["attention_mask"]
        x = self.bert(**x)[0]
        o = self.resnet(b_img).view(batch_size, max_node, -1)
        o = self.trans_obj(o)

        bert_x = x.clone()    # reserve origin bert output (9, 48, 768)
        x = self.trans_txt(x)    # 9, 48, 512
        
        for i in range(self.layer):
            # Text self-attention: batch, max_len, dim
            newx = self.res4mes_x[i](x, self.mhatt_x[i](
                x, x, x, text_mask.unsqueeze(1)))

            # Visual self-attention: batch, 4, odim
            newo = self.res4mes_o[i](o, self.mhatt_o[i](o, o, o, None))

            # Text to Image Gating
            newx_ep = newx.unsqueeze(2).expand(
                batch_size, max_len, objn, newx.size(-1))
            o_ep = newo.unsqueeze(1).expand(batch_size, xn, objn, o.size(-1))
            # batch, xn, objn, dmodel
            x2o_gates = torch.sigmoid(
                self.mhatt_x2o[i](torch.cat((newx_ep, o_ep), -1)))
            x2o = (x2o_gates * inter_matrix * o_ep).sum(2)

            # Image to Text Gating
            x_ep = newx.unsqueeze(1).expand(batch_size, objn, xn, newx.size(-1))
            newo_ep = newo.unsqueeze(2).expand(batch_size, objn, xn, o.size(-1))
            # B O T H
            o2x_gates = torch.sigmoid(
                self.mhatt_o2x[i](torch.cat((x_ep, newo_ep), -1)))
            o2x = (o2x_gates * matrix4obj * x_ep).sum(2)

            newx = self.xgate[i](newx, x2o)
            newo = self.ogate[i](newo, o2x)

            # 9, 48, 512
            x = self.res4ffn_x[i](newx, self.ffn_x[i](newx))
            o = self.res4ffn_o[i](newo, self.ffn_o[i](newo))
        x = torch.cat((bert_x, x), dim=2)
        x = self.hidden2tag(x)
        return -self.crf(x, tags, mask=crf_mask)

    def forward(self, x, b_img, inter_matrix, text_mask, tags):
        """
        inter_matrix: batch, max_len, 4
        text_mask: batch, max_len
        """
        batch_size, xn, objn = inter_matrix.size(0), inter_matrix.size(1), inter_matrix.size(2)
        inter_matrix = inter_matrix.unsqueeze(-1)
        matrix4obj = torch.transpose(inter_matrix, 1, 2)
        crf_mask = x["attention_mask"]
        x = self.bert(**x)[0]
        o = self.resnet(b_img).view(batch_size, max_node, -1)
        o = self.trans_obj(o)

        bert_x = x.clone()    # reserve origin bert output (9, 48, 768)
        x = self.trans_txt(x)
        for i in range(self.layer):
            # Text self-attention: batch, max_len, dim
            newx = self.res4mes_x[i](x, self.mhatt_x[i](
                x, x, x, text_mask.unsqueeze(1)))

            # Visual self-attention: batch, 4, odim
            newo = self.res4mes_o[i](o, self.mhatt_o[i](o, o, o, None))

            # Text to Image Gating
            newx_ep = newx.unsqueeze(2).expand(
                batch_size, max_len, objn, newx.size(-1))
            o_ep = newo.unsqueeze(1).expand(batch_size, xn, objn, o.size(-1))
            # batch, xn, objn, dmodel
            x2o_gates = torch.sigmoid(
                self.mhatt_x2o[i](torch.cat((newx_ep, o_ep), -1)))
            x2o = (x2o_gates * inter_matrix * o_ep).sum(2)

            # Image to Text Gating
            x_ep = newx.unsqueeze(1).expand(batch_size, objn, xn, newx.size(-1))
            newo_ep = newo.unsqueeze(2).expand(batch_size, objn, xn, o.size(-1))
            # B O T H
            o2x_gates = torch.sigmoid(
                self.mhatt_o2x[i](torch.cat((x_ep, newo_ep), -1)))
            o2x = (o2x_gates * matrix4obj * x_ep).sum(2)

            newx = self.xgate[i](newx, x2o)
            newo = self.ogate[i](newo, o2x)

            x = self.res4ffn_x[i](newx, self.ffn_x[i](newx))
            o = self.res4ffn_o[i](newo, self.ffn_o[i](newo))
        x = torch.cat((bert_x, x), dim=2)
        x = self.hidden2tag(x)
        return self.crf.decode(x, mask=crf_mask)


def save_model(model, model_path="./model.pt"):
    torch.save(model.state_dict(), model_path)
    print("Current Best mmner model has beed saved!")



def test(sentence, image):
    time_1 = time.time()
    pdb.set_trace()
    test_dataset = MMNerDataset(sentence=sentence, image=image)
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
    model.eval()
    line = []
    time_2 = time.time()
    with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                b_ntokens = batch["b_ntokens"]

                x = batch["x"]
                b_img = batch["b_img"]
                inter_matrix = batch["b_matrix"]
                text_mask = x["attention_mask"]
                y = batch["y"]
                output = model(x, b_img, inter_matrix, text_mask, y)
                time_3 = time.time()
                # write into file
                for idx, pre_seq in enumerate(output):
                    ground_seq = y[idx]
                    for pos, (pre_idx, ground_idx) in enumerate(zip(pre_seq, ground_seq)):
                        if ground_idx == tag2idx["PAD"] or ground_idx == tag2idx["X"] or ground_idx == tag2idx["CLS"] or ground_idx == tag2idx["SEP"]:
                            continue
                        else:
                            predict_tag = idx2tag[pre_idx] if idx2tag[pre_idx] not in [
                                "PAD", "X", "CLS", "SEP"] else "O"
                            line.append("{} {}\n".format(b_ntokens[idx][pos], predict_tag))
                time_4 = time.time()
                print(f"dataload={time_2 - time_1}, calcuate={time_3 - time_2}, pross={time_4 - time_3}")
    return line
model = MMNerModel().to(device)
model.load_state_dict(torch.load('./model.pt'))          
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def prediect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    if 'sentence' not in request.files:
        return jsonify({'error': 'No sentence provided'}), 400
    lines = test(request.form.get('sentence'),request.files['image'])
    res = {
      'Tokens':[],
      'Entity_type':[]  
    }
    for line in lines:
        temp = line.split()
        res["Tokens"].append(temp[0])
        res["Entity_type"].append(temp[1])
    return jsonify(res)
    #test(request.get_json('sentence'), file)
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8001)
