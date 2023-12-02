# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from itertools import cycle
from multiprocessing import Pool
from sklearn.feature_extraction.text import CountVectorizer as CV
from collections import defaultdict, OrderedDict

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import scipy
import pandas as pd

import sentencepiece as spm

def collate_fn(batch):
    # バッチ内の各要素を分離
    bos = torch.LongTensor([1])
    eos = torch.LongTensor([2])
    labels = [[1, 0] if item['label'] else [0, 1] for item in batch ]
    texts = [torch.cat([bos, torch.LongTensor(item['text']), eos]) for item in batch]
    content_words = [item['cw'] for item in batch]

    #print(texts)

    # テンソルに変換
    labels = torch.tensor(labels, dtype=torch.float, requires_grad=False)

    # パディング
    padded_texts = pad_sequence(texts, batch_first=False, padding_value=3)

    #print(texts)
    #print(labels)
    #print(content_words)

    return padded_texts, labels, content_words

def create_batch_sampler(data, batch_size):
    indices = torch.arange(len(data)).tolist()
    sorted_indices = sorted(indices, key=lambda idx: len(data[idx]["text"]))

    batch_indices = []
    start = 0
    end = min(start + batch_size, len(data))
    while True:
        batch_indices.append(sorted_indices[start: end])

        if end >= len(data):
            break

        start = end
        end = min(start + batch_size, len(data))

    return batch_indices

class CustomDataset(Dataset):
    def __init__(self, data, data_name):
        self.data_df = data.dropna()
        data = self.data_df.to_dict(orient='records')
        self.data = sorted(data, key=lambda x: len(x['text']))
        
        # SentencePiece トークナイザーの初期化
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(f"{data_name}.model")

    def __len__(self):
        return len(self.data)
    
    def text_len(self):
        f_len = lambda x: len(x.split(" "))
        text_len = self.data_df["text"].apply(f_len)
        return text_len.max(), text_len.min(), text_len.mean()

    def __getitem__(self, idx):
        item = self.data[idx]
        label = item['label']
        text = item['text']
        content_words = item['cw']

        # テキストのトークナイズ
        encoded_text_ids = self.tokenizer.EncodeAsIds(text)
        encoded_cw_ids = self.tokenizer.EncodeAsIds(content_words)


        return {
            'label': label,
            'text': encoded_text_ids,
            'cw': encoded_cw_ids
        }



def bow_to_array(bow_raw, count):
    bow = count.fit_transform(bow_raw)
    return bow


class VocabEntry(object):
    def __init__(self, vocab_size=100000, cycle=False):
        super(VocabEntry, self).__init__()
        self.vocab_size = vocab_size

        self.word2id = OrderedDict()
        self.unk_id = 3
        self.word2id["<pad>"] = 0
        self.word2id["<s>"] = 1
        self.word2id["</s>"] = 2
        self.word2id["<unk>"] = self.unk_id
        self.id2word_ = list(self.word2id.keys())
        self.cycle = cycle

        self.is_cw = {}
        self.is_cw[0] = 0
        self.is_cw[1] = 0
        self.is_cw[2] = 0
        self.is_cw[3] = 0

    def create_ft_embed(self, ft_file="data/cc.ja.300.vec"):
        self.ft_embed = np.random.randn(len(self) - 4, 300)
        with open(ft_file) as f:
            for line in f:
                word, vec = line.split(" ", 1)

                wid = self[word]
                if wid > self.unk_id:
                    v = np.fromstring(vec, sep=" ", dtype=np.float32)
                    self.ft_embed[wid - 4, :] = v

        _mu = self.ft_embed.mean()
        _std = self.ft_embed.std()
        self.ft_embed = np.vstack(
            [np.random.randn(4, self.ft_embed.shape[1]) * _std + _mu, self.ft_embed]
        )

    def __getitem__(self, word):
        idx = self.word2id.get(word, self.unk_id)
        return idx if idx < self.vocab_size else self.unk_id

    def __contains__(self, word):
        return word in self.word2id

    def __len__(self):
        return min(len(self.word2id), self.vocab_size)

    def id2word(self, wid):
        return self.id2word_[wid]

    def items(self):
        return {self.id2word(k): k for k in range(len(self))}

    def cw_check(self, wid):
        return self.is_cw[wid]

    def decode_sentence(self, sentence):
        decoded_sentence = []
        for wid_t in sentence:
            wid = wid_t.item()
            decoded_sentence.append(self.id2word_[wid])
        return decoded_sentence

    def build(self, sents):
        wordcount = defaultdict(int)
        for sent in sents:
            for w in sent:
                wordcount[w] += 1
        sorted_words = sorted(wordcount, key=wordcount.get, reverse=True)

        for idx, word in enumerate(sorted_words):
            self.word2id[word] = idx + 4
            self.is_cw[idx + 4] = 0
        self.id2word_ = list(self.word2id.keys())


##################################################################################################################
####   Mono Text Data    #########################################################################################
##################################################################################################################
class MonoTextData(object):
    def __init__(
        self,
        fname,
        label=False,
        max_length=None,
        vocab=None,
        b_vocab=None,
        ft=False,
        check=False,
        attn=False,
        cycle=False,
        fname2=None,
    ):
        self.check = check
        self.attn = attn
        self.cycle = cycle
        if self.cycle:
            super(MonoTextData, self).__init__()
            (
                self.data,
                self.bow,
                self.data_tra,
                self.bow_tra,
                self.vocab,
                self.b_vocab,
                self.dropped,
                self.labels,
            ) = self._read_corpus_cycle(
                fname, fname2, label, max_length, vocab, b_vocab, ft
            )
        else:
            super(MonoTextData, self).__init__()
            (
                self.data,
                self.bow,
                self.vocab,
                self.b_vocab,
                self.dropped,
                self.labels,
            ) = self._read_corpus(fname, label, max_length, vocab, b_vocab, ft)

    def __len__(self):
        return len(self.data)

    def _read_corpus(self, fname, label, max_length, vocab, b_vocab, ft):
        data = []
        bow_raw = []
        labels = [] if label else None
        dropped = 0

        sents = []
        bows = []
        with open(fname) as fin:
            for line in fin:
                if label:
                    split_line = line.strip().split("\t")
                    lb = split_line[0]
                    if len(split_line) > 1:
                        sent_line = split_line[1].split()
                    else:
                        sent_line = ""
                    if len(split_line) < 3:
                        bow_raw_line = ""
                        bow_line = ""

                    else:
                        bow_raw_line = split_line[2]
                        bow_line = bow_raw_line.split()

                else:
                    split_line = line.strip().split("\t")
                    sent_line = split_line[0].split()
                    if len(split_line) < 2:
                        bow_raw_line = ""
                        bow_line = ""
                    else:
                        bow_raw_line = split_line[1]
                        bow_line = bow_raw_line.split()

                if len(sent_line) < 1:
                    dropped += 1
                    continue

                if max_length:
                    if len(sent_line) > max_length:
                        dropped += 1
                        continue

                if label:
                    labels.append(int(lb))

                sents.append(sent_line)
                bows.append(bow_line)
                bow_raw.append(bow_raw_line)
                data.append(sent_line)

        if isinstance(vocab, int):
            vocab = VocabEntry(vocab)
            vocab.build(sents)
            if ft:
                vocab.create_ft_embed()
        elif vocab is None:
            vocab = VocabEntry()
            vocab.build(sents)
            if ft:
                vocab.create_ft_embed()

        data = [[vocab[word] for word in x] for x in data]
        # print(vocab["僕"],vocab["１"],vocab["君"])

        # print(bows)
        if bows != None and not self.attn:
            if isinstance(b_vocab, int):
                b_vocab = VocabEntry(b_vocab)
                b_vocab.build(bows)
                if ft:
                    b_vocab.create_ft_embed()
            elif b_vocab is None:
                b_vocab = VocabEntry()
                b_vocab.build(bows)

            # print(b_vocab["位相"], b_vocab["遅れ"], b_vocab["君"])
            if self.check:
                print(vocab.items())
                print(b_vocab.items())
            count = CV(vocabulary=b_vocab.items())

            # print(count.get_feature_names())
            # results = bow_to_array(bow_raw, count)

            with Pool(16) as pool:
                results = [
                    pool.apply_async(bow_to_array, (bow_raw, count))
                    for (bow_raw, count) in zip(
                        np.array_split(bow_raw, 16), [count] * (16)
                    )
                ]
                bow = [f.get().toarray().tolist() for f in results]
                pool.close()
                pool.terminate()

            bow = sum(bow, [])
            print("bow_process finished")
            # print(len(bow[1]))
            # print(bow[1])

            return data, bow, vocab, b_vocab, dropped, labels

        elif self.attn:

            bow = [[vocab[word] for word in x] for x in bows]
            if self.cycle:
                for bow_word in bow:
                    for bow_id in bow_word:
                        vocab.is_cw[bow_id] = 1

            bow_positions = [
                [i + 1 for word, i in zip(words, range(len(words))) if (word in bows)]
                for words, bows in zip(data, bow)
            ]
            return data, bow_positions, vocab, b_vocab, dropped, labels

    def _read_corpus_cycle(self, fname, fname2, label, max_length, vocab, b_vocab, ft):
        data_ori = []
        bow_raw_ori = []
        data_tra = []
        bow_raw_tra = []
        labels = [] if label else None
        dropped = 0

        sents_ori = []
        bows_ori = []
        sents_tra = []
        bows_tra = []
        with open(fname) as fin, open(fname2) as fin2:
            for ori, tra in zip(fin, fin2):
                if label:
                    split_ori = ori.strip().split("\t")
                    split_tra = tra.strip().split("\t")
                    lb = split_ori[0]
                    if len(split_ori) > 1 and len(split_tra) > 1:
                        sent_ori = split_ori[1].split()
                        sent_tra = split_tra[1].split()
                    else:
                        sent_ori = ""
                        sent_tra = ""

                    if len(split_ori) < 3 or len(split_tra) < 3:
                        bow_raw_ori_line = ""
                        bow_ori_line = ""
                        bow_raw_tra_line = ""
                        bow_tra_line = ""
                    else:
                        bow_raw_ori_line = split_ori[2]
                        bow_ori_line = bow_raw_ori_line.split()
                        bow_raw_tra_line = split_tra[2]
                        bow_tra_line = bow_raw_tra_line.split()

                else:
                    split_ori = ori.strip().split("\t")
                    sent_ori = split_ori[0].split()
                    split_tra = tra.strip().split("\t")
                    sent_tra = split_tra[0].split()
                    if len(split_ori) < 2 or len(split_tra) < 2:
                        bow_raw_ori_line = ""
                        bow_ori_line = ""
                        bow_raw_tra_line = ""
                        bow_tra_line = ""
                    else:
                        bow_raw_ori_line = split_ori[1]
                        bow_ori_line = bow_raw_ori_line.split()
                        bow_raw_tra_line = split_tra[1]
                        bow_tra_line = bow_raw_tra_line.split()

                if len(sent_ori) < 1 or len(sent_tra) < 1:
                    dropped += 1
                    continue

                if max_length:
                    if len(sent_ori) > max_length or len(sent_tra) > max_length:
                        dropped += 1
                        continue

                if label:
                    labels.append(int(lb))

                sents_ori.append(sent_ori)
                bows_ori.append(bow_ori_line)
                bow_raw_ori.append(bow_raw_ori_line)
                data_ori.append(sent_ori)
                sents_tra.append(sent_tra)
                bows_tra.append(bow_tra_line)
                bow_raw_tra.append(bow_raw_tra_line)
                data_tra.append(sent_tra)

        if isinstance(vocab, int):
            vocab = VocabEntry(vocab, cycle=True)
            vocab.build(sents_ori)
            if ft:
                vocab.create_ft_embed()
        elif vocab is None:
            vocab = VocabEntry(cycle=True)
            vocab.build(sents_ori)
            if ft:
                vocab.create_ft_embed()

        data_ori = [[vocab[word] for word in x] for x in data_ori]
        data_tra = [[vocab[word] for word in x] for x in data_tra]
        # print(vocab["僕"],vocab["１"],vocab["君"])

        # print(bows)
        if bows_ori is not None and not self.cycle:
            if isinstance(b_vocab, int):
                b_vocab = VocabEntry(b_vocab, cycle=True)
                b_vocab.build(bows_ori)
                if ft:
                    b_vocab.create_ft_embed()
            elif b_vocab is None:
                b_vocab = VocabEntry(cycle=True)
                b_vocab.build(bows_ori)

            # print(b_vocab["位相"], b_vocab["遅れ"], b_vocab["君"])
            if self.check:
                print(vocab.items())
                print(b_vocab.items())
            count = CV(vocabulary=b_vocab.items())

            # print(count.get_feature_names())
            # results = bow_to_array(bow_raw, count)

            with Pool(16) as pool:
                results = [
                    pool.apply_async(bow_to_array, (bow_raw_ori, count))
                    for (bow_raw_ori, count) in zip(
                        np.array_split(bow_raw_ori, 16), [count] * (16)
                    )
                ]
                bow_ori = [f.get().toarray().tolist() for f in results]
                pool.close()
                pool.terminate()
            bow_ori = sum(bow_ori, [])

            with Pool(16) as pool:
                results = [
                    pool.apply_async(bow_to_array, (bow_raw_tra, count))
                    for (bow_raw_tra, count) in zip(
                        np.array_split(bow_raw_tra, 16), [count] * (16)
                    )
                ]
                bow_tra = [f.get().toarray().tolist() for f in results]
                pool.close()
                pool.terminate()
            bow_tra = sum(bow_tra, [])

            print("bow_process finished")
            # print(len(bow[1]))
            # print(bow[1])

            return data_ori, bow_ori, data_tra, bow_tra, vocab, b_vocab, dropped, labels

        elif self.cycle:
            bow_ori = [[vocab[word] for word in x] for x in bows_ori]
            bow_positions_ori = [
                [i + 1 for word, i in zip(words, range(len(words))) if (word in bows)]
                for words, bows in zip(data_ori, bow_ori)
            ]

            bow_tra = [[vocab[word] for word in x] for x in bows_tra]
            bow_positions_tra = [
                [i + 1 for word, i in zip(words, range(len(words))) if (word in bows)]
                for words, bows in zip(data_tra, bow_tra)
            ]
            for bows in bow_ori:
                for bow_id in bows:
                    vocab.is_cw[bow_id] = 1

            return (
                data_ori,
                bow_positions_ori,
                data_tra,
                bow_positions_tra,
                vocab,
                b_vocab,
                dropped,
                labels,
            )

    def _to_tensor(self, batch_data, batch_first, device, min_len=0, max_len=None):

        batch_data = [sent + [self.vocab["</s>"]] for sent in batch_data]
        sents_len = [len(sent) for sent in batch_data]
        if max_len is not None:
            max_len = max(max(sents_len), max_len)
        else:
            max_len = max(sents_len)
        # print(max_len)
        max_len = max(min_len, max_len)
        batch_size = len(sents_len)
        sents_new = []
        sents_new.append([self.vocab["<s>"]] * batch_size)
        for i in range(max_len):
            sents_new.append(
                [
                    sent[i] if len(sent) > i else self.vocab["<pad>"]
                    for sent in batch_data
                ]
            )
        sents_ts = torch.tensor(
            sents_new, dtype=torch.long, requires_grad=False, device=device
        )

        if batch_first:
            sents_ts = sents_ts.permute(1, 0).contiguous()

        return sents_ts, [length + 1 for length in sents_len]

    def data_iter(self, batch_size, device, batch_first=False, shuffle=True):
        index_arr = np.arange(len(self.data))
        if shuffle:
            np.random.shuffle(index_arr)
        batch_num = int(np.ceil(len(index_arr)) / float(batch_size))
        for i in range(batch_num):
            batch_ids = index_arr[i * batch_size : (i + 1) * batch_size]
            batch_data = [self.data[index] for index in batch_ids]
            batch_data, sents_len = self._to_tensor(batch_data, batch_first, device)
            yield batch_data, sents_len

    def create_data_batch_label(self, batch_size, device, batch_first=False, min_len=5):
        sents_len = np.array([len(sent) for sent in self.data])
        sort_idx = np.argsort(sents_len)
        sort_len = sents_len[sort_idx]

        change_loc = []
        for i in range(1, len(sort_len)):
            if sort_len[i] != sort_len[i - 1]:
                change_loc.append(i)
        change_loc.append(len(sort_len))

        batch_data_list = []
        batch_label_list = []
        total = 0
        cur = 0
        for idx in change_loc:
            while cur < idx:
                batch_data = []
                batch_label = []
                nxt = min(cur + batch_size, idx)
                for id_ in range(cur, nxt):
                    batch_data.append(self.data[sort_idx[id_]])
                    batch_label.append(self.labels[sort_idx[id_]])
                cur = nxt
                batch_data, sents_len = self._to_tensor(
                    batch_data, batch_first, device, min_len
                )
                batch_data_list.append(batch_data)
                batch_label_list.append(batch_label)

                total += batch_data.size(0)
                assert sents_len == ([sents_len[0]] * len(sents_len))

        return batch_data_list, batch_label_list

    def create_data_batch_labels(
        self, batch_size, device, batch_first=False, min_len=5
    ):
        sents_len = np.array([len(sent) for sent in self.data])
        sort_idx = np.argsort(sents_len)
        sort_len = sents_len[sort_idx]

        change_loc = []
        for i in range(1, len(sort_len)):
            if sort_len[i] != sort_len[i - 1]:
                change_loc.append(i)
        change_loc.append(len(sort_len))

        batch_data_list = []
        batch_label_list = []
        total = 0
        cur = 0
        for idx in change_loc:
            while cur < idx:
                batch_data = []
                batch_label = []
                nxt = min(cur + batch_size, idx)
                for id_ in range(cur, nxt):
                    batch_data.append(self.data[sort_idx[id_]])
                    if self.labels[sort_idx[id_]]:
                        batch_label.append([1, 0])
                    else:
                        batch_label.append([0, 1])
                batch_label = torch.tensor(
                    batch_label, dtype=torch.float, requires_grad=False, device=device
                )
                cur = nxt
                batch_data, sents_len = self._to_tensor(
                    batch_data, batch_first, device, min_len
                )
                batch_data_list.append(batch_data)
                batch_label_list.append(batch_label)

                total += batch_data.size(0)
                assert sents_len == ([sents_len[0]] * len(sents_len))

        return batch_data_list, batch_label_list

    def create_data_batch(self, batch_size, device, batch_first=False):
        sents_len = np.array([len(sent) for sent in self.data])
        sort_idx = np.argsort(sents_len)
        sort_len = sents_len[sort_idx]

        change_loc = []
        for i in range(1, len(sort_len)):
            if sort_len[i] != sort_len[i - 1]:
                change_loc.append(i)
        change_loc.append(len(sort_len))

        batch_data_list = []
        total = 0
        cur = 0
        for idx in change_loc:
            while cur < idx:
                batch_data = []
                nxt = min(cur + batch_size, idx)
                for id_ in range(cur, nxt):
                    batch_data.append(self.data[sort_idx[id_]])
                cur = nxt
                batch_data, sents_len = self._to_tensor(batch_data, batch_first, device)
                batch_data_list.append(batch_data)

                total += batch_data.size(0)
                assert sents_len == ([sents_len[0]] * len(sents_len))

        return batch_data_list

    def create_data_batch_feats(self, batch_size, feats, device, batch_first=False):
        sents_len = np.array([len(sent) for sent in self.data])
        print("Maximum length: %d" % max(sents_len))
        sort_idx = np.argsort(sents_len)
        sort_len = sents_len[sort_idx]

        change_loc = []
        for i in range(1, len(sort_len)):
            if sort_len[i] != sort_len[i - 1]:
                change_loc.append(i)
        change_loc.append(len(sort_len))

        batch_data_list = []
        batch_feat_list = []
        total = 0
        cur = 0
        for idx in change_loc:
            while cur < idx:
                batch_data = []
                batch_feat = []
                nxt = min(cur + batch_size, idx)
                for id_ in range(cur, nxt):
                    batch_data.append(self.data[sort_idx[id_]])
                    batch_feat.append(feats[sort_idx[id_]])
                cur = nxt
                batch_data, sents_len = self._to_tensor(batch_data, batch_first, device)
                batch_data_list.append(batch_data)
                batch_feat = torch.tensor(
                    batch_feat, dtype=torch.float, requires_grad=False, device=device
                )
                batch_feat_list.append(batch_feat)

                total += batch_data.size(0)
                assert sents_len == ([sents_len[0]] * len(sents_len))

        return batch_data_list, batch_feat_list

    def create_data_batch_bow(self, batch_size, feats, device, batch_first=False):
        sents_len = np.array([len(sent) for sent in self.data])
        print("Maximum length: %d" % max(sents_len))
        sort_idx = np.argsort(sents_len)
        sort_len = sents_len[sort_idx]

        change_loc = []
        for i in range(1, len(sort_len)):
            if sort_len[i] != sort_len[i - 1]:
                change_loc.append(i)
        change_loc.append(len(sort_len))

        batch_data_list = []
        batch_feat_list = []
        batch_bow_list = []
        total = 0
        cur = 0
        for idx in change_loc:
            while cur < idx:
                batch_data = []
                batch_feat = []
                batch_bow = []
                nxt = min(cur + batch_size, idx)
                for id_ in range(cur, nxt):
                    batch_data.append(self.data[sort_idx[id_]])
                    batch_feat.append(feats[sort_idx[id_]])
                    batch_bow.append(self.bow[sort_idx[id_]])
                cur = nxt
                batch_data, sents_len = self._to_tensor(batch_data, batch_first, device)
                batch_data_list.append(batch_data)
                batch_feat = torch.tensor(
                    batch_feat, dtype=torch.float, requires_grad=False, device=device
                )
                batch_feat_list.append(batch_feat)
                batch_bow = torch.tensor(
                    batch_bow, dtype=torch.float, requires_grad=False, device=device
                )
                batch_bow_list.append(batch_bow)

                total += batch_data.size(0)
                assert sents_len == ([sents_len[0]] * len(sents_len))

        return batch_data_list, batch_feat_list, batch_bow_list

    def create_data_batch_cbow(
        self, batch_size, device, logging=None, batch_first=False, min_len=5
    ):
        sents_len = np.array([len(sent) for sent in self.data])
        if logging is not None:
            logging("Maximum length: {}".format(max(sents_len)))
        else:
            print("Maximum length: %d" % max(sents_len))
        sort_idx = np.argsort(sents_len)
        sort_len = sents_len[sort_idx]

        change_loc = []
        for i in range(1, len(sort_len)):
            if sort_len[i] != sort_len[i - 1]:
                change_loc.append(i)
        change_loc.append(len(sort_len))

        batch_data_list = []
        batch_label_list = []
        batch_bow_list = []
        total = 0
        cur = 0
        for idx in change_loc:
            while cur < idx:
                batch_data = []
                batch_label = []
                batch_bow = []
                nxt = min(cur + batch_size, idx)
                for id_ in range(cur, nxt):
                    batch_data.append(self.data[sort_idx[id_]])
                    if self.labels[sort_idx[id_]]:
                        batch_label.append([1, 0])
                    else:
                        batch_label.append([0, 1])
                    batch_bow.append(self.bow[sort_idx[id_]])
                batch_label = torch.tensor(
                    batch_label, dtype=torch.float, requires_grad=False, device=device
                )
                cur = nxt
                batch_data, sents_len = self._to_tensor(
                    batch_data, batch_first, device, min_len
                )
                batch_data_list.append(batch_data)
                batch_label_list.append(batch_label)
                batch_bow = torch.tensor(
                    batch_bow, dtype=torch.float, requires_grad=False, device=device
                ).to_sparse()
                batch_bow_list.append(batch_bow)

                total += batch_data.size(0)
                assert sents_len == ([sents_len[0]] * len(sents_len))

        return batch_data_list, batch_label_list, batch_bow_list

    def create_data_batch_cattn(
        self,
        batch_size,
        device,
        logging=None,
        batch_first=False,
        min_len=5,
    ):
        sents_len = np.array([len(sent) for sent in self.data])
        if logging is not None:
            logging("Maximum length: {}".format(max(sents_len)))
        else:
            print("Maximum length: %d" % max(sents_len))
        sort_idx = np.argsort(sents_len)
        sort_len = sents_len[sort_idx]

        change_loc = []
        for i in range(1, len(sort_len)):
            if sort_len[i] != sort_len[i - 1]:
                change_loc.append(i)
        change_loc.append(len(sort_len))

        batch_data_list = []
        batch_label_list = []
        batch_bow_list = []
        total = 0
        cur = 0
        for idx in change_loc:
            while cur < idx:
                batch_data = []
                batch_label = []
                batch_bow = []
                nxt = min(cur + batch_size, idx)
                for id_ in range(cur, nxt):
                    batch_data.append(self.data[sort_idx[id_]])
                    if self.labels[sort_idx[id_]]:
                        batch_label.append([1, 0])
                    else:
                        batch_label.append([0, 1])
                    batch_bow.append(self.bow[sort_idx[id_]])
                batch_label = torch.tensor(
                    batch_label, dtype=torch.float, requires_grad=False, device=device
                )
                cur = nxt
                batch_data, sents_len = self._to_tensor(
                    batch_data, batch_first, device, min_len
                )
                batch_data_list.append(batch_data)
                batch_label_list.append(batch_label)
                batch_bow_list.append(batch_bow)

                total += batch_data.size(0)
                assert sents_len == ([sents_len[0]] * len(sents_len))

        return batch_data_list, batch_label_list, batch_bow_list

    def create_data_batch_cattn_ft(
        self,
        batch_size,
        device,
        logging=None,
        batch_first=False,
        min_len=5,
    ):
        data_tapple = list(
            zip(
                self.data[: int(len(self.data) / 2)],
                self.data[int(len(self.data) / 2) :],
            )
        )
        data_tapple.extend(
            list(
                zip(
                    self.data[int(len(self.data) / 2) :],
                    self.data[: int(len(self.data) / 2)],
                )
            )
        )

        sents_len = np.array([max(len(sent[0]), len(sent[1])) for sent in data_tapple])
        if logging is not None:
            logging("Maximum length: {}".format(max(sents_len)))
        else:
            print("Maximum length: %d" % max(sents_len))
        sort_idx = np.argsort(sents_len)
        sort_len = sents_len[sort_idx]

        change_loc = []
        for i in range(1, len(sort_len)):
            if sort_len[i] != sort_len[i - 1]:
                change_loc.append(i)
        change_loc.append(len(sort_len))

        batch_data_list = []
        batch_label_list = []
        batch_tlabel_list = []
        batch_bow_list = []
        batch_tar_list = []
        total = 0
        cur = 0
        for idx in change_loc:
            while cur < idx:
                batch_data = []
                batch_label = []
                batch_tlabel = []
                batch_bow = []
                batch_tar = []
                nxt = min(cur + batch_size, idx)
                for id_ in range(cur, nxt):
                    batch_data.append(data_tapple[sort_idx[id_]][0])
                    batch_tar.append(data_tapple[sort_idx[id_]][1])
                    if self.labels[sort_idx[id_]]:
                        batch_label.append([1, 0])
                        batch_tlabel.append([0, 1])
                    else:
                        batch_label.append([0, 1])
                        batch_tlabel.append([1, 0])
                    batch_bow.append(self.bow[sort_idx[id_]])
                batch_label = torch.tensor(
                    batch_label, dtype=torch.float, requires_grad=False, device=device
                )
                batch_tlabel = torch.tensor(
                    batch_tlabel, dtype=torch.float, requires_grad=False, device=device
                )
                cur = nxt
                # print(f"sort_len[id_]:{sort_len[id_]}")
                batch_data, sents_len = self._to_tensor(
                    batch_data, batch_first, device, min_len, sort_len[id_] + 1
                )
                # print(f"batch_data:{batch_data}")

                batch_tar, tar_sents_len = self._to_tensor(
                    batch_tar, batch_first, device, min_len, sort_len[id_] + 1
                )
                # print(f"batch_tar:{batch_tar}")
                batch_data_list.append(batch_data)
                batch_label_list.append(batch_label)
                batch_tlabel_list.append(batch_tlabel)
                batch_bow_list.append(batch_bow)
                batch_tar_list.append(batch_tar)

                total += batch_data.size(0)
                # print(f"sents_len:{sents_len}")
                # print(f"tar_sents_len:{tar_sents_len}")
                # assert sents_len == ([sents_len[0]] * len(sents_len))

        return (
            batch_data_list,
            batch_label_list,
            batch_tlabel_list,
            batch_bow_list,
            batch_tar_list,
        )

    def create_data_batch_cycle(
        self,
        batch_size,
        device,
        logging=None,
        batch_first=False,
        min_len=5,
    ):
        sents_len = np.array([len(sent) for sent in self.data])
        if logging is not None:
            logging("Maximum length: {}".format(max(sents_len)))
        else:
            print("Maximum length: %d" % max(sents_len))
        sort_idx = np.argsort(sents_len)
        sort_len = sents_len[sort_idx]

        change_loc = []
        for i in range(1, len(sort_len)):
            if sort_len[i] != sort_len[i - 1]:
                change_loc.append(i)
        change_loc.append(len(sort_len))

        batch_data_list = []
        batch_tdata_list = []
        batch_label_list = []
        batch_bow_list = []
        batch_tbow_list = []
        total = 0
        cur = 0
        for idx in change_loc:
            while cur < idx:
                batch_data = []
                batch_tdata = []
                batch_label = []
                batch_bow = []
                batch_tbow = []
                nxt = min(cur + batch_size, idx)
                for id_ in range(cur, nxt):
                    batch_data.append(self.data[sort_idx[id_]])
                    batch_tdata.append(self.data_tra[sort_idx[id_]])
                    if self.labels[sort_idx[id_]]:
                        batch_label.append([1, 0])
                    else:
                        batch_label.append([0, 1])
                    batch_bow.append(self.bow[sort_idx[id_]])
                    batch_tbow.append(self.bow_tra[sort_idx[id_]])
                batch_label = torch.tensor(
                    batch_label, dtype=torch.float, requires_grad=False, device=device
                )

                cur = nxt
                batch_data, sents_len = self._to_tensor(
                    batch_data, batch_first, device, min_len
                )
                batch_tdata, _ = self._to_tensor(
                    batch_tdata, batch_first, device, min_len
                )

                batch_data_list.append(batch_data)
                batch_tdata_list.append(batch_tdata)
                batch_label_list.append(batch_label)
                batch_bow_list.append(batch_bow)
                batch_tbow_list.append(batch_tbow)

                total += batch_data.size(0)
                assert sents_len == ([sents_len[0]] * len(sents_len))

        """
        print([[self.vocab.id2word(id) for id in ids] for ids in batch_data_list[1]])
        print([[self.vocab.id2word(id) for id in ids] for ids in batch_tdata_list[1]])

        print(self.bow_tra[:4])

        exit()
        # """

        return (
            batch_data_list,
            batch_label_list,
            batch_bow_list,
            batch_tdata_list,
            batch_tbow_list,
        )

    def data_sample(self, nsample, device, batch_first=False, shuffle=True):
        index_arr = np.arange(len(self.data))
        if shuffle:
            np.random.shuffle(index_arr)
        batch_ids = index_arr[:nsample]
        batch_data = [self.data[index] for index in batch_ids]

        batch_data, sents_len = self._to_tensor(batch_data, batch_first, device)

        return batch_data, sents_len
