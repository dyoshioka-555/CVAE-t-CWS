# Copyright (c) 2020-present, Royal Bank of Canada.
# Copyright (c) 2020-present, Juxian He
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#################################################################################################
# Code is based on the VAE lagging encoder (https://arxiv.org/abs/1901.05534) implementation
# from https://github.com/jxhe/vae-lagging-encoder by Junxian He
#################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
from .utils import log_sum_exp
import numpy as np
from scipy.stats import ortho_group


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length, attn_weight):
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length
        self.attn_w = attn_weight

    def eval(self, alpha=1.0):
        reward = 0
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class, dropout):
        super(MLP, self).__init__()
        assert isinstance(input_dims, int), "Invalid type for input_dims!"
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)

        for i, n_hidden in enumerate(n_hiddens):
            l_i = i + 1
            layers["fc{}".format(l_i)] = nn.Linear(current_dims, n_hidden)
            layers["relu{}".format(l_i)] = nn.ReLU()
            layers["drop{}".format(l_i)] = nn.Dropout(dropout)
            current_dims = n_hidden
        layers["out"] = nn.Linear(current_dims, n_class)

        self.model = nn.Sequential(layers)

    def forward(self, input):
        return self.model.forward(input)


class AddPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.device = device
        positional_encoding_weight = self._initialize_weight().to(device)
        self.register_buffer("positional_encoding_weight", positional_encoding_weight)

    def forward(self, x):
        # print(x.size())
        seq_len, batch_size, _ = x.size()
        # print(self.positional_encoding_weight[:seq_len, :].unsqueeze(1).size())
        return x + self.positional_encoding_weight[:seq_len, :].unsqueeze(1).expand(
            seq_len, batch_size, -1
        )

    def _get_positional_encoding(self, pos, i):
        w = pos / (10000 ** (((2 * i) // 2) / self.d_model))
        if i % 2 == 0:
            return np.sin(w)
        else:
            return np.cos(w)

    def _initialize_weight(self):
        positional_encoding_weight = [
            [self._get_positional_encoding(pos, i) for i in range(1, self.d_model + 1)]
            for pos in range(1, self.max_len + 1)
        ]
        return torch.tensor(positional_encoding_weight).float()


class GaussianEncoderBase(nn.Module):
    def __init__(self):
        super(GaussianEncoderBase, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def sample(self, inputs, nsamples):
        mu, logvar = self.forward(inputs)
        z = self.reparameterize(mu, logvar, nsamples)
        return z, (mu, logvar)

    def encode(self, inputs, nsamples=1):
        mu, logvar = self.forward(inputs)
        z = self.reparameterize(mu, logvar, nsamples)
        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(1)
        return z, KL

    def reparameterize(self, mu, logvar, nsamples=1):
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(0).expand(nsamples, batch_size, nz)
        std_expd = std.unsqueeze(0).expand(nsamples, batch_size, nz)

        eps = torch.zeros_like(std_expd).normal_()
        return mu_expd + torch.mul(eps, std_expd)

    def sample_from_inference(self, x, nsamples=1):
        mu, logvar = self.forward(x)
        batch_size, nz = mu.size()
        return mu.unsqueeze(0).expand(nsamples, batch_size, nz)

    def eval_inference_dist(self, x, z, param=None):
        nz = z.size(2)
        if not param:
            mu, logvar = self.forward(x)
        else:
            mu, logvar = param

        mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        var = logvar.exp()
        dev = z - mu

        log_density = -0.5 * ((dev**2) / var).sum(dim=-1) - 0.5 * (
            nz * math.log(2 * math.pi) + logvar.sum(-1)
        )

        return log_density.squeeze(0)

    def calc_mi(self, x, label):
        mu, logvar = self.forward(x, label)

        x_batch, nz = mu.size()

        neg_entropy = (
            -0.5 * nz * math.log(2 * math.pi) - 0.5 * (1 + logvar).sum(-1)
        ).mean()

        z_samples = self.reparameterize(mu, logvar, 1)

        mu, logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
        var = logvar.exp()

        dev = z_samples - mu

        log_density = -0.5 * ((dev**2) / var).sum(dim=-1) - 0.5 * (
            nz * math.log(2 * math.pi) + logvar.sum(-1)
        )

        log_qz = log_sum_exp(log_density, dim=0) - math.log(x_batch)

        return (neg_entropy - log_qz.mean(-1)).item()


class TransformerEncoder(GaussianEncoderBase):
    def __init__(
        self,
        ni,
        nh,
        nz,
        label_ni,
        label_nz,
        vocab_size,
        model_init,
        emb_init,
        device,
        nhead=2,
        num_layers=2,
    ):
        super(TransformerEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, ni)
        self.pos_embed = AddPositionalEncoding(ni, 64, device)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=ni, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

        self.linear = nn.Linear(nh, 2 * nz, bias=False)
        # self.label_linear = nn.Linear(label_ni, nh, bias=False)
        self.emb_liner = nn.Linear(ni, nh, bias=False)

        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def forward(self, inputs, label, bow=None, shift=None):
        seq_len, batch_size = inputs.size()

        if len(inputs.size()) > 2:
            word_embed = torch.matmul(inputs, self.embed.weight)
        else:
            word_embed = self.embed(inputs)

        hs = self.transformer_encoder(word_embed)
        hs = self.emb_liner(hs)
        # print(hs.size())

        hidden_repr = hs.mean(dim=0)
        # print(hidden_repr.size())
        mean, logvar = self.linear(hidden_repr).chunk(2, -1)
        # print(mean, logvar)
        # exit()

        if bow is not None:
            word_embed = self.pos_embed(word_embed)
            hs = self.emb_liner(word_embed)

            zero = torch.zeros_like(hs[0][0])

            bow = [[i - shift for i in b if i - shift >= 0] for b in bow]
            for p, j in zip(bow, range(batch_size)):
                for k in range(seq_len):
                    if k not in p:
                        hs[k][j] = zero

            return mean, logvar, hs

        return mean, logvar

    def encode(self, inputs, label, bow, shift, nsamples=1):
        mu, logvar, hs = self.forward(inputs, label, bow, shift)
        z = self.reparameterize(mu, logvar, nsamples)
        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(1)
        return z, KL, hs


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        ni,
        nh,
        nz,
        label_nz,
        dropout_in,
        dropout_out,
        vocab,
        model_init,
        emb_init,
        device,
        num_layers=1,
    ):
        super(TransformerDecoder, self).__init__()

        self.nz = nz
        self.nh = nh
        self.vocab = vocab
        self.device = device

        self.embed = nn.Embedding(len(vocab), ni, padding_idx=-1)
        self.pos_embed = AddPositionalEncoding(ni, 64, device)

        self.dropout_in = nn.Dropout(dropout_in)
        self.dropout_out = nn.Dropout(dropout_out)

        self.label_linear = nn.Linear(2, label_nz, bias=False)

        # Transformer decoder layer
        self.decoder_layer = nn.TransformerDecoderLayer(ni, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=num_layers
        )

        self.trans_linear = nn.Linear(nz, ni, bias=False)
        self.output_linear = nn.Linear(ni, nh, bias=False)
        self.pred_linear = nn.Linear(nh * 2, len(vocab), bias=False)

        self.softmax = nn.Softmax(dim=1)
        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def forward(self, inputs, z, label, hs):
        z2 = self.label_linear(label).unsqueeze(0)
        z = torch.cat([z, z2], -1)

        n_sample, batch_size, _ = z.size()
        seq_len = inputs.size(0)

        word_embed = self.embed(inputs)
        word_embed = self.dropout_in(word_embed)

        z = self.trans_linear(z)  # shape: 1 x batch_size x ni
        z = z.repeat(seq_len, 1, 1)  # repeat along sequence length dimension
        memory = z

        output = self.transformer_decoder(word_embed, memory)
        output = self.output_linear(output)

        # print(output.size(), hs.size())

        output = torch.transpose(output, 0, 1)

        hs = torch.transpose(hs, 0, 1)
        t_output = torch.transpose(output, 1, 2)

        s = torch.bmm(hs, t_output)

        attention_weight = self.softmax(s)

        c = torch.zeros(batch_size, 1, self.nh, device=self.device)

        for i in range(attention_weight.size()[2]):  # 10回ループ
            # attention_weight[:,:,i].size() = ([100, 29])
            # i番目のGRU層に対するattention weightを取り出すが、テンソルのサイズをhsと揃えるためにunsqueezeする
            unsq_weight = attention_weight[:, :, i].unsqueeze(
                2
            )  # unsq_weight.size() = ([100, 29, 1])

            # hsの各ベクトルをattention weightで重み付けする
            weighted_hs = hs * unsq_weight  # weighted_hs.size() = ([100, 29, 128])

            # attention weightで重み付けされた各hsのベクトルをすべて足し合わせてコンテキストベクトルを作成
            weight_sum = torch.sum(weighted_hs, axis=1).unsqueeze(
                1
            )  # weight_sum.size() = ([100, 1, 128])

            c = torch.cat([c, weight_sum], dim=1)  # c.size() = ([100, i, 128])

        # 箱として用意したzero要素が残っているのでスライスして削除
        c = c[:, 1:, :]

        output = torch.cat([output, c], dim=2)
        output = torch.transpose(output, 0, 1)
        output = self.dropout_out(output)

        output_logits = self.pred_linear(output)
        # print(output_logits.size())

        return output_logits

    def decode(self, z, label, hs, max_length=50):
        batch_size = z.size(0)

        decoded_batch = [[] for _ in range(batch_size)]

        decoder_input = torch.tensor(
            [self.vocab["<s>"]] * batch_size, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        end_symbol = torch.tensor(
            [self.vocab["</s>"]] * batch_size, dtype=torch.long, device=self.device
        )

        mask = torch.ones((batch_size), dtype=torch.uint8, device=self.device)

        for _ in range(max_length):
            word_embed = self.embed(decoder_input)
            logits = self.forward(word_embed, z.unsqueeze(0), label, hs)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_word = torch.argmax(probs, dim=-1, keepdim=True)
            decoder_input = torch.cat([decoder_input, next_word], dim=0)

            for i in range(batch_size):
                if mask[i].item():
                    decoded_batch[i].append(self.vocab.DecodeIds(next_word[i].item()))

            mask = torch.mul((next_word.squeeze(0) != end_symbol), mask)

        return decoded_batch


class LSTMEncoder(GaussianEncoderBase):
    def __init__(
        self, ni, nh, nz, label_ni, label_nz, vocab_size, model_init, emb_init, device
    ):
        super(LSTMEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, ni)
        self.pos_embed = AddPositionalEncoding(ni, 64, device)
        self.nh = nh
        self.label_ni = label_ni

        self.lstm = nn.LSTM(
            input_size=ni, hidden_size=nh, num_layers=2, bidirectional=True
        )
        self.linear = nn.Linear(nh, 2 * nz, bias=False)
        self.label_linear = nn.Linear(label_ni, nh, bias=False)
        self.emb_liner = nn.Linear(ni, nh, bias=False)

        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def forward(self, inputs, label, bow=None, shift=None):
        seq_len, batch_size = inputs.size()

        if len(inputs.size()) > 2:
            word_embed = torch.matmul(inputs, self.embed.weight)
        else:
            word_embed = self.embed(inputs)

        hs, (last_state, last_cell) = self.lstm(
            word_embed,
        )  # (h_init, c_init)
        seq_len, bsz, hidden_size = hs.size()
        hidden_repr = hs.view(seq_len, bsz, 2, -1).mean(2)
        hidden_repr = torch.max(hidden_repr, 0)[0]

        mean, logvar = self.linear(hidden_repr).chunk(2, -1)

        if bow is not None:
            ###Positional embedding###
            word_embed = self.pos_embed(word_embed)
            hs = self.emb_liner(word_embed)

            zero = torch.zeros_like(hs[0][0])

            bow = [[i - shift for i in b if i - shift >= 0] for b in bow]
            for p, j in zip(bow, range(bsz)):
                for k in range(seq_len):
                    if k not in p:
                        hs[k][j] = zero

            return mean, logvar, hs

        return mean, logvar

    def encode(self, inputs, label, bow, shift, nsamples=1):
        mu, logvar, hs = self.forward(inputs, label, bow, shift)
        z = self.reparameterize(mu, logvar, nsamples)
        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(1)
        return z, KL, hs


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        ni,
        nh,
        nz,
        label_nz,
        dropout_in,
        dropout_out,
        vocab,
        model_init,
        emb_init,
        device,
    ):
        super(LSTMDecoder, self).__init__()
        self.nz = nz
        self.nh = nh
        self.vocab = vocab
        self.device = device
        self.bow_dim = 256
        self.bow = None

        self.embed = nn.Embedding(len(vocab), ni, padding_idx=-1)

        self.dropout_in = nn.Dropout(dropout_in)
        self.dropout_out = nn.Dropout(dropout_out)

        self.label_linear = nn.Linear(2, label_nz, bias=False)

        self.trans_linear = nn.Linear(nz, nh, bias=False)

        self.softmax = nn.Softmax(dim=1)

        self.lstm = nn.LSTM(input_size=ni + nz, hidden_size=nh, num_layers=1)

        self.pred_linear = nn.Linear(nh * 2, len(vocab), bias=False)

        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def forward(self, inputs, z, label, hs):
        z2 = self.label_linear(label).unsqueeze(0)
        z = torch.cat([z, z2], -1)

        n_sample, batch_size, _ = z.size()
        seq_len = inputs.size(0)

        word_embed = self.embed(inputs)
        word_embed = self.dropout_in(word_embed)

        if n_sample == 1:
            z_ = z.expand(seq_len, batch_size, self.nz)
        else:
            raise NotImplementedError

        word_embed = torch.cat((word_embed, z_), -1)

        z = z.view(batch_size * n_sample, self.nz)
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)
        output, _ = self.lstm(word_embed, (h_init, c_init))

        output = torch.transpose(output, 0, 1)

        hs = torch.transpose(hs, 0, 1)
        t_output = torch.transpose(output, 1, 2)

        s = torch.bmm(hs, t_output)

        attention_weight = self.softmax(s)

        c = torch.zeros(batch_size, 1, self.nh, device=self.device)

        for i in range(attention_weight.size()[2]):  # 10回ループ
            # attention_weight[:,:,i].size() = ([100, 29])
            # i番目のGRU層に対するattention weightを取り出すが、テンソルのサイズをhsと揃えるためにunsqueezeする
            unsq_weight = attention_weight[:, :, i].unsqueeze(
                2
            )  # unsq_weight.size() = ([100, 29, 1])

            # hsの各ベクトルをattention weightで重み付けする
            weighted_hs = hs * unsq_weight  # weighted_hs.size() = ([100, 29, 128])

            # attention weightで重み付けされた各hsのベクトルをすべて足し合わせてコンテキストベクトルを作成
            weight_sum = torch.sum(weighted_hs, axis=1).unsqueeze(
                1
            )  # weight_sum.size() = ([100, 1, 128])

            c = torch.cat([c, weight_sum], dim=1)  # c.size() = ([100, i, 128])

        # 箱として用意したzero要素が残っているのでスライスして削除
        c = c[:, 1:, :]

        output = torch.cat([output, c], dim=2)
        output = torch.transpose(output, 0, 1)
        output = self.dropout_out(output)
        output_logits = self.pred_linear(output)

        return output_logits.view(-1, batch_size, len(self.vocab))

    def decode(self, z, greedy=True):
        # print("decodeが使われたYO")
        batch_size = z.size(0)
        decoded_batch = [[] for _ in range(batch_size)]

        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)

        decoder_hidden = (h_init, c_init)
        decoder_input = torch.tensor(
            [self.vocab["<s>"]] * batch_size, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        end_symbol = torch.tensor(
            [self.vocab["</s>"]] * batch_size, dtype=torch.long, device=self.device
        )

        mask = torch.ones((batch_size), dtype=torch.uint8, device=self.device)
        length_c = 1
        while mask.sum().item() != 0 and length_c < 100:
            word_embed = self.embed(decoder_input)
            word_embed = torch.cat((word_embed, z.unsqueeze(0)), dim=-1)

            output, decoder_hidden = self.lstm(word_embed, decoder_hidden)

            decoder_output = self.pred_linear(output)
            output_logits = decoder_output.squeeze(0)

            if greedy:
                select_index = torch.argmax(output_logits, dim=1)
            else:
                sample_prob = F.softmax(output_logits, dim=1)
                select_index = torch.multinomial(sample_prob, num_samples=1).squeeze(1)

            decoder_input = select_index.unsqueeze(0)
            length_c += 1

            for i in range(batch_size):
                if mask[i].item():
                    decoded_batch[i].append(
                        self.vocab.DecodeIds(select_index[i].item())
                    )

            mask = torch.mul((select_index != end_symbol), mask)

        return decoded_batch

    def beam_search_decode(self, z1, z2, hs, K=5, max_t=20):
        decoded_batch = []
        attn_w_batch = []
        z2 = self.label_linear(z2)
        z = torch.cat([z1, z2], -1)

        batch_size, nz = z.size()

        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)

        for idx in range(batch_size):
            decoder_input = torch.tensor(
                [[self.vocab["<s>"]]], dtype=torch.long, device=self.device
            )
            decoder_hidden = (
                h_init[:, idx, :].unsqueeze(1),
                c_init[:, idx, :].unsqueeze(1),
            )
            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0.1, 1, None)
            live_hypotheses = [node]

            completed_hypotheses = []

            t = 0
            flag = 1
            while len(completed_hypotheses) < K and t < max_t:
                t += 1

                # print(t)
                # print([node.wordid for node in live_hypotheses])
                decoder_input = torch.cat(
                    [node.wordid for node in live_hypotheses], dim=1
                )

                decoder_hidden_h = torch.cat(
                    [node.h[0] for node in live_hypotheses], dim=1
                )
                decoder_hidden_c = torch.cat(
                    [node.h[1] for node in live_hypotheses], dim=1
                )

                decoder_hidden = (decoder_hidden_h, decoder_hidden_c)

                word_embed = self.embed(decoder_input)
                # word_embed = self.pos_embed(word_embed)

                word_embed = torch.cat(
                    (
                        word_embed,
                        z[idx].view(1, 1, -1).expand(1, len(live_hypotheses), nz),
                    ),
                    dim=-1,
                )

                output, decoder_hidden = self.lstm(word_embed, decoder_hidden)
                # print(f"word_embed:{word_embed.size()},out:{output.size()}")
                output = torch.transpose(output, 0, 1)
                _hs = torch.transpose(hs, 0, 1)
                _hs = _hs[idx].unsqueeze(0)
                _hs = torch.cat([_hs for _ in range(len(live_hypotheses))], dim=0)
                t_output = torch.transpose(output, 1, 2)
                # print(f"_hs:{_hs.size()},t_out:{t_output.size()}")

                s = torch.bmm(_hs, t_output)

                attention_weight = self.softmax(s)

                c = torch.zeros(len(live_hypotheses), 1, self.nh, device=self.device)

                for i in range(attention_weight.size()[2]):
                    # attention_weight[:,:,i].size() = ([100, 29])
                    # i番目のGRU層に対するattention weightを取り出すが、テンソルのサイズをhsと揃えるためにunsqueezeする
                    unsq_weight = attention_weight[:, :, i].unsqueeze(
                        2
                    )  # unsq_weight.size() = ([100, 29, 1])

                    # hsの各ベクトルをattention weightで重み付けする
                    weighted_hs = (
                        _hs * unsq_weight
                    )  # weighted_hs.size() = ([100, 29, 128])

                    # attention weightで重み付けされた各hsのベクトルをすべて足し合わせてコンテキストベクトルを作成
                    weight_sum = torch.sum(weighted_hs, axis=1).unsqueeze(
                        1
                    )  # weight_sum.size() = ([100, 1, 128])

                    c = torch.cat([c, weight_sum], dim=1)  # c.size() = ([100, i, 128])

                # 箱として用意したzero要素が残っているのでスライスして削除
                c = c[:, 1:, :]

                output = torch.cat([output, c], dim=2)
                output = torch.transpose(output, 0, 1)

                output_logits = self.pred_linear(output)
                decoder_output = F.log_softmax(output_logits, dim=-1)

                prev_logp = torch.tensor(
                    [node.logp for node in live_hypotheses],
                    dtype=torch.float,
                    device=self.device,
                )
                decoder_output = decoder_output + prev_logp.view(
                    1, len(live_hypotheses), 1
                )

                decoder_output = decoder_output.view(-1)

                log_prob, indexes = torch.topk(
                    decoder_output, K - len(completed_hypotheses)
                )

                live_ids = indexes // len(self.vocab)
                word_ids = indexes % len(self.vocab)

                if flag:
                    attention_weight = torch.cat([attention_weight for _ in range(K)])
                    # print(attention_weight.size())
                    flag = 0

                live_hypotheses_new = []
                for live_id, word_id, log_prob_, attn_w in zip(
                    live_ids, word_ids, log_prob, attention_weight
                ):
                    node = BeamSearchNode(
                        (
                            decoder_hidden[0][:, live_id, :].unsqueeze(1),
                            decoder_hidden[1][:, live_id, :].unsqueeze(1),
                        ),
                        live_hypotheses[live_id],
                        word_id.view(1, 1),
                        log_prob_,
                        t,
                        attn_w,
                    )

                    if word_id.item() == self.vocab["</s>"]:
                        completed_hypotheses.append(node)
                    else:
                        live_hypotheses_new.append(node)

                live_hypotheses = live_hypotheses_new

                if len(completed_hypotheses) == K:
                    break

            for live in live_hypotheses:
                completed_hypotheses.append(live)

            utterances = []
            attn_w_list = []
            for n in sorted(
                completed_hypotheses, key=lambda node: node.logp, reverse=True
            ):
                utterance = []
                attn_weight = []
                utterance.append(self.vocab.DecodeIds(n.wordid.item()))
                attn_weight.append(n.attn_w)
                while n.prevNode is not None:
                    n = n.prevNode
                    utterance.append(self.vocab.DecodeIds(n.wordid.item()))
                    attn_weight.append(n.attn_w)

                utterance = utterance[::-1]
                utterances.append(utterance)
                attn_weight = torch.cat([i.unsqueeze(0) for i in attn_weight[::-1][1:]])

                break

            decoded_batch.append(utterances[0])
            attn_w_batch.append(attn_weight)

        return decoded_batch, attn_w_batch
