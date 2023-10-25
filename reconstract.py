# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import config_attn as config
import torch
from utils.text_utils import MonoTextData
from models_tf.decomposed_vae import attnVAE
import argparse
import os
from utils.dist_utils import cal_log_density

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(font="IPAGothic")

import pandas as pd


def get_coordinates(a, b, p):
    pa = p - a
    ba = b - a
    t = torch.sum(pa * ba) / torch.sum(ba * ba)
    d = torch.norm(pa - t * ba, 2)
    return t, d


def main(args):
    conf = config.CONFIG[args.data_name]
    data_pth = "data/%s" % args.data_name
    train_data_pth = os.path.join(data_pth, "train_data.txt")

    if args.ft:
        vocab = pd.read_pickle(os.path.join(args.load_path, "vocab.pkl"))
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

    dev_data_pth = os.path.join(data_pth, "dev_data.txt")
    dev_data = MonoTextData(
        dev_data_pth, True, vocab=vocab, b_vocab=b_vocab, attn=args.attn
    )

    test_file_name = args.test_file if args.test_file else "test_data.txt"
    test_data_pth = os.path.join(data_pth, test_file_name)

    if args.vtrain:
        test_data = train_data
    elif args.dev:
        test_data = dev_data
    else:
        test_data = MonoTextData(
            test_data_pth, True, vocab=vocab, b_vocab=b_vocab, attn=args.attn
        )

    if args.wo_unk:
        with open(data_pth + "/test_data_wo_unk.txt", "w") as f:
            for data, label in zip(test_data.data, test_data.labels):
                data = [vocab.id2word(x) for x in data]
                f.write(str(label) + "\t" + " ".join(data) + "\n")
        exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.bow or args.attn:
        if args.ft:
            kwargs = {
                "train": ([1], None, None, None, None),
                "valid": (None, None, None, None, None),
                "test": (None, None, None, None, None),
                "bsz": 32,
                "save_path": args.load_path,
                "logging": None,
                "use_bow": args.bow,
                "debug_nan": False,
                # "fine_tuning": args.ft,
            }
        else:
            kwargs = {
                "train": ([1], None, None),
                "valid": (None, None, None),
                "test": (None, None, None),
                "bsz": 32,
                "save_path": args.load_path,
                "logging": None,
                "use_bow": args.bow,
                "debug_nan": False,
                # "fine_tuning": False,
            }
    else:
        kwargs = {
            "train": ([1], None),
            "valid": (None, None),
            "test": (None, None),
            "bsz": 32,
            "save_path": args.load_path,
            "logging": None,
            "use_bow": args.bow,
            "debug_nan": False,
        }
    params = conf["params"]
    params["vae_params"]["vocab"] = vocab
    params["vae_params"]["device"] = device
    params["vae_params"]["label_ni"] = 2

    len_bow = 0
    if args.bow:
        len_bow = len(b_vocab)
    params["vae_params"]["bow_size"] = len_bow
    kwargs = dict(kwargs, **params)
    
    model = attnVAE(**kwargs)
    model.load(args.load_path)
    model.vae.eval()

    sep_id = -1
    for idx, x in enumerate(test_data.labels):
        if x == 1:
            sep_id = idx
            break

    bsz = 64
    if args.out_name is not None:
        out_file_name = args.out_name
    elif args.visualize:
        out_file_name = "empty.txt"

    elif args.trans:
        out_file_name = "transfer_results.txt"
    else:
        out_file_name = "reconstracted_results.txt"
    if args.vtrain:
        out_file_name = out_file_name.replace("_results.txt", "_train_data.txt")
    if args.dev:
        out_file_name = out_file_name.replace("_results.txt", "_dev_data.txt")
    with open(os.path.join(args.load_path, out_file_name), "w") as f_label, open(
        os.path.join(args.load_path, "_" + out_file_name), "w"
    ) as f_no_label:
        if args.visualize:
            idx = 0
            label = test_data.labels[idx]
            _idx = idx + bsz

            input_text, _ = test_data._to_tensor(
                test_data.data[idx:_idx], batch_first=False, device=device
            )

            if not args.trans:
                feat = torch.tensor(
                    [1, 0] if label else [0, 1],
                    dtype=torch.float,
                    requires_grad=False,
                    device=device,
                ).expand(_idx - idx, 2)
            else:
                feat = torch.tensor(
                    [0, 1] if label else [1, 0],
                    dtype=torch.float,
                    requires_grad=False,
                    device=device,
                ).expand(_idx - idx, 2)
            z1, _, hs = model.vae.lstm_encoder(
                input_text[: min(input_text.shape[0], 25)],
                feat,
                test_data.bow[idx:_idx],
                0,
            )
            texts, attn_w = model.vae.decoder.decode(z1, feat, hs)
            if args.vtrain:
                dir_name = "attn_visuals_train"
            elif args.dev:
                dir_name = "attn_visuals_dev"
            else:
                dir_name = "attn_visuals"
            os.makedirs(args.load_path + dir_name, exist_ok=True)
            # print(input_text)
            # print(input_text.size())
            for i in range(args.num):
                with torch.no_grad():
                    # print(attn_w[i])
                    columns = [vocab.id2word(j) for j in input_text.transpose(0, 1)[i]]
                    index = [j for j in texts[i][1:]]
                    # print(columns)
                    # print(index)
                    df = pd.DataFrame(
                        data=torch.transpose(attn_w[i], 0, 1)
                        .squeeze(-1)
                        .transpose(0, 1)
                        .cpu()
                        .numpy(),
                        columns=columns,
                        index=index,
                    )
                    if "<pad>" in list(df.columns):
                        df.drop(columns="<pad>", inplace=True)
                    if args.trans:
                        df.to_csv(
                            args.load_path
                            + dir_name
                            + f"/attn_w_visualize_trans{i}.csv",
                            encoding="utf-8",
                        )
                    else:
                        df.to_csv(
                            args.load_path + dir_name + f"/attn_w_visualize{i}.csv",
                            encoding="utf-8",
                        )

            if args.data_name == "toda_lecture":
                idx = 848
            else:
                idx = 4585 if not args.vtrain else 213201
            label = test_data.labels[idx]
            _idx = idx + bsz

            input_text, _ = test_data._to_tensor(
                test_data.data[idx:_idx], batch_first=False, device=device
            )

            if not args.trans:
                feat = torch.tensor(
                    [1, 0] if label else [0, 1],
                    dtype=torch.float,
                    requires_grad=False,
                    device=device,
                ).expand(_idx - idx, 2)
            else:
                feat = torch.tensor(
                    [0, 1] if label else [1, 0],
                    dtype=torch.float,
                    requires_grad=False,
                    device=device,
                ).expand(_idx - idx, 2)
            z1, _, hs = model.vae.lstm_encoder(
                input_text[: min(input_text.shape[0], 25)],
                feat,
                test_data.bow[idx:_idx],
                0,
            )

            texts, attn_w = model.vae.decoder.decode(z1, feat, hs)
            os.makedirs(args.load_path + f"attn_visuals_train", exist_ok=True)
            # print(input_text)
            # print(input_text.size())
            for i in range(args.num):
                with torch.no_grad():
                    # print(attn_w[i])
                    columns = [vocab.id2word(j) for j in input_text.transpose(0, 1)[i]]
                    index = [j for j in texts[i][1:]]
                    # print(columns)
                    # print(index)
                    df = pd.DataFrame(
                        data=torch.transpose(attn_w[i], 0, 1)
                        .squeeze(-1)
                        .transpose(0, 1)
                        .cpu()
                        .numpy(),
                        columns=columns,
                        index=index,
                    )
                    if "<pad>" in list(df.columns):
                        df.drop(columns="<pad>", inplace=True)
                    if args.trans:
                        df.to_csv(
                            args.load_path
                            + dir_name
                            + f"/attn_w_visualize_trans{i+args.num}.csv",
                            encoding="utf-8",
                        )
                    else:
                        df.to_csv(
                            args.load_path
                            + dir_name
                            + f"/attn_w_visualize{i+args.num}.csv",
                            encoding="utf-8",
                        )

        else:
            idx = 0
            step = 0
            n_samples = len(test_data.labels)
            while idx < n_samples:
                label = test_data.labels[idx]
                # print(label)
                _idx = idx + bsz if label else min(idx + bsz, sep_id)
                _idx = min(_idx, n_samples)
                # var_id = neg_idx if label else pos_idx
                input_text, _ = test_data._to_tensor(
                    test_data.data[idx:_idx], batch_first=False, device=device
                )

                if not args.trans:
                    feat = torch.tensor(
                        [1, 0] if label else [0, 1],
                        dtype=torch.float,
                        requires_grad=False,
                        device=device,
                    ).expand(_idx - idx, 2)
                else:
                    feat = torch.tensor(
                        [0, 1] if label else [1, 0],
                        dtype=torch.float,
                        requires_grad=False,
                        device=device,
                    ).expand(_idx - idx, 2)

                if args.bow:
                    bow = torch.tensor(
                        test_data.bow[idx:_idx],
                        dtype=torch.float,
                        requires_grad=False,
                        device=device,
                    )
                if args.attn:
                    z1, _, hs = model.vae.lstm_encoder(
                        input_text[: min(input_text.shape[0], 25)],
                        feat,
                        test_data.bow[idx:_idx],
                        0,
                    )
                else:
                    z1, _ = model.vae.lstm_encoder(
                        input_text[: min(input_text.shape[0], 10)], feat
                    )
                # tra_z2 = model.vae.mlp_encoder.var_embedding[var_id:var_id + 1, :].expand(
                #    _idx - idx, -1)

                texts, attn_w = model.vae.decoder.decode(z1, feat, hs)

                for text in texts:
                    if args.trans:
                        f_label.write(
                            "%d\t%s\n" % (int(not label), " ".join(text[1:-1]))
                        )
                    else:
                        f_label.write("%d\t%s\n" % (label, " ".join(text[1:-1])))
                    f_no_label.write("%s\n" % " ".join(text[1:-1]))

                idx = _idx
                step += 1
                if step % 50 == 0:
                    print(step, idx)


def add_args(parser):
    parser.add_argument("--data_name", type=str, default="toda_lecture")
    parser.add_argument("--out_name", type=str, default=None)
    parser.add_argument("--test_file", "-t", type=str, default=None)
    parser.add_argument("--load_path", type=str)
    parser.add_argument("--bow", default=False, action="store_true")
    parser.add_argument("--trans", default=False, action="store_true")
    parser.add_argument("--attn", default=False, action="store_true")
    parser.add_argument("--ft", default=False, action="store_true")
    parser.add_argument("--visualize", "-v", default=False, action="store_true")
    parser.add_argument("--vtrain", default=False, action="store_true")
    parser.add_argument("--dev", default=False, action="store_true")
    parser.add_argument("--wo_unk", "-u", default=False, action="store_true")
    parser.add_argument("--num", "-n", type=int, default=32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
