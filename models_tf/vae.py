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

from .utils import uniform_initializer, value_initializer, gumbel_softmax
from .base_network import LSTMEncoder, LSTMDecoder, TransformerEncoder, TransformerDecoder


class DecomposedVAE(nn.Module):
    def __init__(
        self,
        lstm_ni,
        lstm_nh,
        lstm_nz,
        label_ni,
        label_nz,
        dec_ni,
        dec_nh,
        dec_dropout_in,
        dec_dropout_out,
        vocab,
        label_num,
        device,
        bow_size,
    ):
        super(DecomposedVAE, self).__init__()
        model_init = uniform_initializer(0.01)
        enc_embed_init = uniform_initializer(0.1)
        dec_embed_init = uniform_initializer(0.1)
        self.device = device
        self.vocab = vocab
        self.lstm_encoder = TransformerEncoder(
            lstm_ni,
            lstm_nh,
            lstm_nz,
            label_ni,
            label_nz,
            len(vocab),
            model_init,
            enc_embed_init,
            device,
        )
        self.decoder = LSTMDecoder(
            dec_ni,
            dec_nh,
            lstm_nz + label_nz,
            dec_dropout_in,
            dec_dropout_out,
            vocab,
            model_init,
            dec_embed_init,
            device,
            bow_size,
        )

    def encode_syntax(self, x, label, bow, shift, nsamples=1):
        return self.lstm_encoder.encode(x, label, bow, shift, nsamples)

    def decode(self, x, z, label, bow):
        return self.decoder(x, z, label, bow)

    def loss(self, x, label, bow=None, shift=None, tau=1.0, nsamples=1, no_ic=True):

        z1, KL1, hs = self.encode_syntax(x, label, bow, shift, nsamples)
        # _, batch_size, _ = z1.size()
        outputs = self.decode(x[:-1], z1, label, hs) #.view(-1, batch_size, len(self.vocab))

        if no_ic:
            reg_ic = torch.zeros(10)
        else:
            soft_outputs = gumbel_softmax(outputs, tau)
            log_density = self.lstm_encoder.eval_inference_dist(soft_outputs, z1)
            logit = log_density.exp()
            reg_ic = -torch.log(torch.sigmoid(logit))
        return outputs, KL1, reg_ic

    def calc_mi_q(self, x, label):
        mi = self.lstm_encoder.calc_mi(x, label)
        return mi
