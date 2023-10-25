# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from utils.exp_utils import create_exp_dir
from utils.text_utils import MonoTextData
import argparse
import os
import torch
import time
import config_tf
from models_tf.decomposed_vae import attnVAE

import pandas as pd


def main(args):
    conf = config_tf.CONFIG[args.data_name]
    data_pth = "data/%s" % args.data_name
    train_data_pth = os.path.join(data_pth, "train_data.txt")

    if args.fine_tuning and not args.debug:
        vocab = pd.read_pickle(os.path.join(args.load_path, "vocab.pkl"))
        if args.bow:
            b_vocab = pd.read_pickle(os.path.join(args.load_path, "b_vocab.pkl"))
        else:
            b_vocab = None
    elif os.path.isfile(os.path.join(data_pth, "vocab.pkl")):
        vocab = pd.read_pickle(os.path.join(data_pth, "vocab.pkl"))
        b_vocab = pd.read_pickle(os.path.join(data_pth, "b_vocab.pkl"))
    else:
        vocab = None
        b_vocab = None

    train_data = MonoTextData(
        train_data_pth, True, vocab=vocab, b_vocab=b_vocab, attn=args.attn
    )

    vocab = train_data.vocab

    print("Vocabulary size: %d" % len(vocab))
    if args.bow:
        b_vocab = train_data.b_vocab
        print("bow_Vocabulary size: %d" % len(b_vocab))
    else:
        b_vocab = None

    if not args.debug:
        dev_data_pth = os.path.join(data_pth, "dev_data.txt")
        dev_data = MonoTextData(
            dev_data_pth, True, vocab=vocab, b_vocab=b_vocab, attn=args.attn
        )
        test_data_pth = os.path.join(data_pth, "test_data.txt")
        test_data = MonoTextData(
            test_data_pth, True, vocab=vocab, b_vocab=b_vocab, attn=args.attn
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.attn:
        save_path = "{}-{}-attn+pe".format(args.save, args.data_name)
    elif args.bow:
        save_path = "{}-{}-bow+pe".format(args.save, args.data_name)
    else:
        save_path = "{}-{}".format(args.save, args.data_name)
    save_path = os.path.join(save_path, time.strftime("%Y%m%d-%H%M%S"))
    scripts_to_save = [
        "run.py",
        "models_tf/decomposed_vae.py",
        "models_tf/vae.py",
        "models_tf/base_network.py",
        "config_tf.py",
    ]

    logging = create_exp_dir(
        save_path, scripts_to_save=scripts_to_save, debug=args.debug
    )
    logging("Vocabulary size: {}".format(len(vocab)))
    if args.bow:
        logging("b_Vocabulary size: {}".format(len(b_vocab)))

    if not args.fine_tuning and not args.debug:
        pd.to_pickle(vocab, os.path.join(save_path, "vocab.pkl"))
        if args.bow or args.attn:
            pd.to_pickle(b_vocab, os.path.join(save_path, "b_vocab.pkl"))

    if args.bow:
        train = train_data.create_data_batch_cbow(args.bsz, device, logging)
        if args.debug_nan or args.debug:
            dev = None
            test = None
        else:
            dev = dev_data.create_data_batch_cbow(args.bsz, device, logging)
            test = test_data.create_data_batch_cbow(args.bsz, device, logging)

    elif args.attn:
        if args.fine_tuning:
            train = train_data.create_data_batch_cattn_ft(args.bsz, device, logging)
        else:
            train = train_data.create_data_batch_cattn(args.bsz, device, logging)
        if args.debug_nan or args.debug:
            dev = None
            test = None
        else:
            if args.fine_tuning:
                dev = dev_data.create_data_batch_cattn_ft(args.bsz, device, logging)
                test = test_data.create_data_batch_cattn_ft(args.bsz, device, logging)
            else:
                dev = dev_data.create_data_batch_cattn(args.bsz, device, logging)
                test = test_data.create_data_batch_cattn(args.bsz, device, logging)

    else:
        train = train_data.create_data_batch_labels(args.bsz, device)
        dev = dev_data.create_data_batch_labels(args.bsz, device)
        test = test_data.create_data_batch_labels(args.bsz, device)

    kwargs = {
        "train": train,
        "valid": dev,
        "test": test,
        "bsz": args.bsz,
        "save_path": save_path,
        "logging": logging,
        "use_bow": args.bow,
        "debug_nan": args.debug_nan,
        "debug": args.debug,
        # "fine_tuning": args.fine_tuning,
    }
    params = conf["params"]
    params["vae_params"]["vocab"] = vocab
    params["vae_params"]["device"] = device
    params["vae_params"]["label_ni"] = 2

    len_bow = len(b_vocab) if args.bow else 0

    params["vae_params"]["bow_size"] = len_bow
    kwargs = dict(kwargs, **params)

    model = attnVAE(**kwargs)

    if args.fine_tuning:
        model.load(args.load_path)

    try:
        valid_loss = model.fit()
        logging("val loss : {}".format(valid_loss))
    except KeyboardInterrupt:
        logging("Exiting from training early")

    model.load(save_path)

    if args.fine_tuning:
        if args.bow or args.attn:
            test_loss = model.evaluate(
                model.test_data,
                model.test_label,
                model.test_tlabel,
                model.test_bow,
                model.test_target,
            )
        else:
            test_loss = model.evaluate(model.test_data, model.test_label)
    else:
        if args.bow or args.attn:
            test_loss = model.evaluate(
                model.test_data, model.test_label, model.test_bow
            )
        else:
            test_loss = model.evaluate(model.test_data, model.test_label)

    logging("test loss: {}".format(test_loss[0]))
    logging("test recon: {}".format(test_loss[1]))
    logging("test kl: {}".format(test_loss[2]))
    if args.fine_tuning:
        logging("test trans: {}".format(test_loss[3]))
        logging("test k2: {}".format(test_loss[4]))


def add_args(parser):
    parser.add_argument(
        "--data_name", type=str, default="toda_lecture", help="data name"
    )
    parser.add_argument(
        "--save", type=str, default="checkpoint/ours", help="directory name to save"
    )
    parser.add_argument("--bsz", type=int, default=32, help="batch size for training")
    parser.add_argument(
        "--debug", default=False, action="store_true", help="enable debug mode"
    )
    parser.add_argument(
        "--bow", default=False, action="store_true", help="using bow feats"
    )
    parser.add_argument(
        "--vocab_check",
        default=False,
        action="store_true",
        help="checking vocabulary size",
    )
    parser.add_argument(
        "--debug_nan",
        default=False,
        action="store_true",
        help="checking vocabulary size",
    )
    parser.add_argument(
        "--attn",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--fine_tuning",
        "-f",
        default=False,
        action="store_true",
    )
    parser.add_argument("--load_path", type=str, default=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
