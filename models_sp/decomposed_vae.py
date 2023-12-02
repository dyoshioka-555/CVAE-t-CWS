# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from utils.exp_utils import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .vae import DecomposedVAE as VAE
from utils.text_utils import collate_fn, create_batch_sampler
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import math
import random


class attnVAE:
    def __init__(
        self,
        train,
        valid,
        test,
        bsz,
        save_path,
        logging,
        log_interval,
        num_epochs,
        enc_lr,
        dec_lr,
        warm_up,
        kl_start,
        beta1,
        ic_weight,
        aggressive,
        debug,
        vae_params,
    ):
        super(attnVAE, self).__init__()
        self.bsz = bsz
        self.save_path = save_path
        self.logging = logging
        self.log_interval = log_interval
        self.num_epochs = num_epochs
        self.enc_lr = enc_lr
        self.dec_lr = dec_lr
        self.warm_up = warm_up
        self.kl_weight = kl_start
        self.beta1 = beta1
        self.ic_weight = ic_weight
        self.aggressive = aggressive
        self.opt_dict = {"not_improved": 0, "lr": 1.0, "best_loss": 1e4}
        self.pre_mi = 0
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug
        self.vocab = vae_params["vocab"]

        """
        for text, label, cw in train:
            print(text, label, cw)
            exit()
        """

        self.train_data = train
        if not debug:
            self.valid_data = DataLoader(valid, batch_sampler=create_batch_sampler(valid.data, self.bsz), collate_fn=collate_fn, num_workers=2, pin_memory=True)
            self.test_data = DataLoader(test, batch_sampler=create_batch_sampler(test.data, self.bsz), collate_fn=collate_fn, num_workers=2, pin_memory=True)

        self.vae = VAE(**vae_params)
        if self.use_cuda:
            self.vae.cuda()

        self.enc_params = list(self.vae.lstm_encoder.parameters())
        self.enc_optimizer = optim.Adam(self.enc_params, lr=self.enc_lr)
        self.dec_optimizer = optim.SGD(self.vae.decoder.parameters(), lr=self.dec_lr)

        self.nbatch = len(self.train_data.data)
        self.anneal_rate = (1.0 - kl_start) / (warm_up * self.nbatch)

    def train(self, epoch):
        self.vae.train()

        total_rec_loss = 0
        total_kl1_loss = 0
        start_time = time.time()
        step = 0
        num_words = 0
        num_sents = 0

        batch_sampler = create_batch_sampler(self.train_data.data, self.bsz)
        random.shuffle(batch_sampler)
        # print("Batch_sampler Done")

        train = DataLoader(self.train_data, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=2, pin_memory=True)
        # print("DataLoader Done")

        for text, label, cw in train:
            batch_data = text.to(self.device, non_blocking=True)
            # print(batch_data)

            batch_label = label.to(self.device, non_blocking=True)

            batch_cw = cw

            _sent_len, batch_size = batch_data.size()
            # print(batch_data)
            # print(f"sent_len:{_sent_len}, batch_size:{batch_size}")

            shift = np.random.randint(max(1, _sent_len - 10))
            batch_data = batch_data[shift:_sent_len, :]
            # batch_data = batch_data[shift : min(_sent_len, shift + 16), :]
            # print(batch_data)
            sent_len, batch_size = batch_data.size()

            if 0:
                if shift > 0:
                    print(
                        f"shift:{shift}, sent_len:{sent_len}, batch_size:{batch_size}"
                    )
            # print(batch_data[-1].tolist())

            """
            if _sent_len > sent_len + shift and (0 not in batch_data[-1].tolist()):
                EOS = torch.tensor(
                    [2 for _ in range(batch_size)],
                    dtype=torch.long,
                    device=self.device,
                )
                batch_data = torch.cat([batch_data, EOS.unsqueeze(0)], dim=0)
                # print(batch_data)
            """
            sent_len, batch_size = batch_data.size()

            target = batch_data[1:]

            # print(target)
            # print(target.size())
            num_words += (sent_len - 1) * batch_size
            num_sents += batch_size
            self.kl_weight = min(1.0, self.kl_weight + self.anneal_rate)
            beta1 = self.beta1 if self.beta1 else self.kl_weight

            loss = 0

            self.enc_optimizer.zero_grad()
            self.dec_optimizer.zero_grad()

            vae_logits, vae_kl1_loss, reg_ic = self.vae.loss(
                batch_data,
                batch_label,
                batch_cw,
                shift,
                no_ic=self.ic_weight == 0,
            )

            vae_logits = vae_logits.view(-1, vae_logits.size(2))
            """
            print(
                f"vae_logits:{torch.argmax(vae_logits,1)}\ntarget.view(-1):{target.view(-1)}"
            )
            """
            vae_rec_loss = F.cross_entropy(
                vae_logits, target.view(-1), reduction="none"
            )

            vae_rec_loss = vae_rec_loss.view(-1, batch_size).sum(0)
            vae_loss = vae_rec_loss + beta1 * vae_kl1_loss
            # print(vae_loss)

            if 0:
                if shift > 0:
                    source_words = [
                        " ".join([self.vocab.DecodeIds(id) for id in ids])
                        for ids in batch_data[:-1].transpose(0, 1).tolist()
                    ]

                    target_words = [
                        " ".join([self.vocab.DecodeIds(id) for id in ids])
                        for ids in target[:-1].transpose(0, 1).tolist()
                    ]

                    for s, b, t in zip(source_words, batch_cw, target_words):
                        self.logging(f"source:{s}, cw:{b}, target:{t}")
                    exit()

            if self.ic_weight > 0:
                vae_loss += self.ic_weight * reg_ic
            vae_loss = vae_loss.mean()
            total_rec_loss += vae_rec_loss.sum().item()
            total_kl1_loss += vae_kl1_loss.sum().item()

            loss = loss + vae_loss

            loss.backward()

            nn.utils.clip_grad_norm_(self.vae.parameters(), 5.0)
            if not self.aggressive:
                self.enc_optimizer.step()
            self.dec_optimizer.step()

            if step % self.log_interval == 0 and step > 0:
                cur_rec_loss = total_rec_loss / num_sents
                cur_kl1_loss = total_kl1_loss / num_sents

                cur_vae_loss = cur_rec_loss + cur_kl1_loss
                elapsed = time.time() - start_time
                self.logging(
                    "| epoch {:2d} | {:5d}/{:5d} batches | {:5.2f} ms/batch | loss {:3.2f} | "
                    "recon {:3.2f} | kl {:3.2f} |".format(
                        epoch,
                        step,
                        self.nbatch,
                        elapsed * 1000 / self.log_interval,
                        cur_vae_loss,
                        cur_rec_loss,
                        cur_kl1_loss,
                    )
                )
                total_rec_loss = 0
                total_kl1_loss = 0
                num_sents = 0
                num_words = 0
                start_time = time.time()
            step += 1

    def evaluate(self, eval_data):
        self.vae.eval()

        total_rec_loss = 0
        total_kl1_loss = 0

        total_mi1 = 0
        num_sents = 0
        num_words = 0

        with torch.no_grad():

            for batch_data, batch_label, batch_cw in eval_data:
                _sent_len, batch_size = batch_data.size()

                shift = np.random.randint(max(1, _sent_len - 10))
                batch_data = batch_data[shift:_sent_len, :]
                # batch_data = batch_data[shift : min(_sent_len, shift + 16), :]
                sent_len, batch_size = batch_data.size()
                """
                if _sent_len > sent_len + shift and (
                    0 not in batch_data[-1].tolist()
                ):
                    EOS = torch.tensor(
                        [2 for _ in range(batch_size)],
                        dtype=torch.long,
                        device=self.device,
                    )
                    batch_data = torch.cat([batch_data, EOS.unsqueeze(0)], dim=0)
                """
                target = batch_data[1:]

                num_sents += batch_size
                num_words += (sent_len - 1) * batch_size

                vae_logits, vae_kl1_loss, _ = self.vae.loss(
                    batch_data, batch_label, batch_cw, shift
                )
                vae_logits = vae_logits.view(-1, vae_logits.size(2))
                vae_rec_loss = F.cross_entropy(
                    vae_logits, target.view(-1), reduction="none"
                )
                total_rec_loss += vae_rec_loss.sum().item()
                total_kl1_loss += vae_kl1_loss.sum().item()

                mi1 = self.vae.calc_mi_q(batch_data, batch_label)
                total_mi1 += mi1 * batch_size

        cur_rec_loss = total_rec_loss / num_sents
        cur_kl1_loss = total_kl1_loss / num_sents
        cur_mi1 = total_mi1 / num_sents
        cur_vae_loss = cur_rec_loss + cur_kl1_loss

        return (
            cur_vae_loss,
            cur_rec_loss,
            cur_kl1_loss,
            cur_mi1,
        )

    def fit(self):
        best_loss = 1e4
        decay_cnt = 0
        for epoch in range(1, self.num_epochs + 1):
            epoch_start_time = time.time()
            self.train(epoch)

            val_loss = self.evaluate(self.valid_data)

            vae_loss = val_loss[1]

            if vae_loss < best_loss:
                self.save(self.save_path)
                best_loss = vae_loss

            if vae_loss > self.opt_dict["best_loss"]:
                self.opt_dict["not_improved"] += 1
                if math.isnan(vae_loss):
                    self.opt_dict["not_improved"] = 0
                    self.load(self.save_path)
                    self.dec_optimizer = optim.SGD(
                        self.vae.decoder.parameters(), lr=self.opt_dict["lr"]
                    )

                if self.opt_dict["not_improved"] == 1 and epoch >= 15:
                    self.opt_dict["lr"] = self.opt_dict["lr"] * 0.8
                    self.load(self.save_path)
                    self.dec_optimizer = optim.SGD(
                        self.vae.decoder.parameters(), lr=self.opt_dict["lr"]
                    )

                if self.opt_dict["not_improved"] >= 2 and epoch >= 15:
                    self.opt_dict["not_improved"] = 0
                    self.opt_dict["lr"] = self.opt_dict["lr"] * 0.5
                    self.load(self.save_path)
                    decay_cnt += 1
                    self.dec_optimizer = optim.SGD(
                        self.vae.decoder.parameters(), lr=self.opt_dict["lr"]
                    )
            else:
                self.opt_dict["not_improved"] = 0
                self.opt_dict["best_loss"] = vae_loss

            if decay_cnt == 10:
                break

            self.logging("-" * 75)
            self.logging(
                "| end of epoch {:2d} | time {:5.2f}s | "
                "kl_weight {:.2f} | vae_lr {:.2f} | loss {:3.2f}".format(
                    epoch,
                    (time.time() - epoch_start_time),
                    self.kl_weight,
                    self.opt_dict["lr"],
                    val_loss[0],
                )
            )
            self.logging(
                "| recon {:3.2f} | kl {:3.2f} | mi {:3.2f}".format(
                    val_loss[1], val_loss[2], val_loss[3]
                )
            )
            self.logging("-" * 75)

        return best_loss

    def save(self, path):
        self.logging("saving to %s" % path)
        model_path = os.path.join(path, "model.pt")
        torch.save(self.vae.state_dict(), model_path)

    def load(self, path):
        model_path = os.path.join(path, "model.pt")
        self.vae.load_state_dict(torch.load(model_path))
