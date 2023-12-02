# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from utils.exp_utils import create_exp_dir
from utils.text_utils import CustomDataset, collate_fn, create_batch_sampler
import argparse
import os
import torch
import time
import config_sp as config
from models_sp.decomposed_vae import attnVAE

import pandas as pd


def main(args):
    conf = config.CONFIG[args.data_name]
    data_pth = "data/%s" % args.data_name
    train_data_pth = os.path.join(data_pth, "train_data.txt")
    train_data = pd.read_csv(train_data_pth, names=["label", "text", "cw"], sep="\t")
    train = CustomDataset(train_data, args.data_name)

    if args.debug:
        dev = None
        test = None
    else:
        dev_data_pth = os.path.join(data_pth, "dev_data.txt")
        dev_data = pd.read_csv(dev_data_pth, names=["label", "text", "cw"], sep="\t")
        dev = CustomDataset(dev_data, args.data_name)

        test_data_pth = os.path.join(data_pth, "test_data.txt")
        test_data = pd.read_csv(test_data_pth, names=["label", "text", "cw"], sep="\t")
        test = CustomDataset(test_data, args.data_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path = "{}-{}-sp".format(args.save, args.data_name)
    save_path = os.path.join(save_path, time.strftime("%Y%m%d-%H%M%S"))
    scripts_to_save = [
        "run_sp.py",
        "models_sp/decomposed_vae.py",
        "models_sp/vae.py",
        "models_sp/base_network.py",
        "config_sp.py",
    ]

    logging = create_exp_dir(
        save_path, scripts_to_save=scripts_to_save, debug=args.debug
    )
    logging("Vocabulary size: {}".format(train.tokenizer.GetPieceSize()))
    logging(f"Train MAX, MIN, AVE: {train.text_len()}")
    if not args.debug:
        logging(f"Dev MAX, MIN, AVE: {dev.text_len()}")
        logging(f"Test MAX, MIN, AVE: {test.text_len()}")

    kwargs = {
        "train": train,
        "valid": dev,
        "test": test,
        "bsz": args.bsz,
        "save_path": save_path,
        "logging": logging,
        "debug": args.debug,
        # "fine_tuning": args.fine_tuning,
    }
    params = conf["params"]
    params["vae_params"]["vocab"] = train.tokenizer
    params["vae_params"]["device"] = device
    params["vae_params"]["label_ni"] = 2

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
        if args.attn:
            test_loss = model.evaluate(
                model.test_data,
                model.test_label,
                model.test_tlabel,
                model.test_bow,
                model.test_target,
            )
        else:
            test_loss = model.evaluate(model.test_data)
    else:
        test_loss = model.evaluate(model.test_data)

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
        "--vocab_check",
        default=False,
        action="store_true",
        help="checking vocabulary size",
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
