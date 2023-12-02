# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pandas as pd
import argparse
import numpy as np
from utils.bleu import compute_bleu
from utils.text_utils import MonoTextData
import torch
from classifier import CNNClassifier, evaluate
import os
# from rouge import Rouge


def main(args):
    data_pth = "data/%s" % args.data_name
    if args.vocab:
        if os.path.isfile(os.path.join(data_pth, "vocab.pkl")):
            vocab = pd.read_pickle(os.path.join(data_pth, "vocab.pkl"))
        else:
            if args.file is not None:
                train_pth = args.load_path + args.file
            else:
                train_pth = os.path.join(data_pth, "train_data.txt")
            train_data = MonoTextData(train_pth, True, vocab=100000)
            vocab = train_data.vocab

        vocab_dict = vocab.items()
        # print(vocab_dict)

        with open(data_pth + f"/vocab_{args.data_name}.txt", "w") as f:
            for k in vocab_dict:
                f.write(k + "\n")
        exit()

    if args.file is not None:
        target_pth = args.load_path + args.file
    elif args.ac_wo and args.trans:
        target_pth = args.load_path + "trans_wo_bow.txt"
    elif args.ac_wo:
        target_pth = args.load_path + "recon_wo_bow.txt"
    elif args.rouge and args.trans:
        target_pth = args.load_path + "trans_bow.txt"
    elif args.rouge:
        target_pth = args.load_path + "recon_bow.txt"
    elif args.trans:
        target_pth = args.load_path + "transfer_results.txt"
    else:
        target_pth = args.load_path + "reconstracted_results.txt"
    if args.train:
        target_pth = target_pth.replace("results.txt", "train_data.txt")

    if args.avg_len:
        with open(target_pth, "r") as f:
            tokens = [
                len(line.split("\t")[1].strip().split(" ")) for line in f.readlines()
            ]
        with open(target_pth, "r") as f:
            lens = [
                len(line.split("\t")[1].strip().replace(" ", ""))
                for line in f.readlines()
            ]
        print(
            f"token\tmean:{np.mean(tokens):.2f}, max:{np.max(tokens)}, min:{np.min(tokens)}"
        )
        print(f"len\tmean:{np.mean(lens):.2f}, max:{np.max(lens)}, min:{np.min(lens)}")
        exit()

    

    if args.b_vocab:
        # if os.path.isfile(os.path.join(args.load_path, "vocab.pkl")):
        #    vocab = pd.read_pickle(os.path.join(args.load_path, "vocab.pkl"))
        # else:
        if args.file is not None:
            train_pth = args.load_path + args.file
        else:
            train_pth = os.path.join(data_pth, "train_data_kansai.txt")
        train_data = MonoTextData(train_pth, True, vocab=100000)
        vocab = train_data.b_vocab

        vocab_dict = vocab.items()
        # print(vocab_dict)

        with open(data_pth + f"/b_vocab_{args.data_name}.txt", "w") as f:
            for k in vocab_dict:
                f.write(k + "\n")
        exit()

    # Classification Accuracy
    if args.ac:
        train_pth = os.path.join(data_pth, "train_data.txt")
        # """
        if os.path.isfile(os.path.join(args.load_path, "vocab.pkl")) and os.path.isfile(
            os.path.join(args.load_path, "b_vocab.pkl")
        ):
            vocab = pd.read_pickle(os.path.join(args.load_path, "vocab.pkl"))
            b_vocab = pd.read_pickle(os.path.join(args.load_path, "b_vocab.pkl"))

        else:
            train_data = MonoTextData(train_pth, True, vocab=100000)
            vocab = train_data.vocab
            b_vocab = train_data.b_vocab
        """
        if os.path.isfile(
            os.path.join(args.load_path, "vocab_wo.pkl")
        ) and os.path.isfile(os.path.join(args.load_path, "b_vocab_wo.pkl")):
            vocab = pd.read_pickle(os.path.join(args.load_path, "vocab_wo.pkl"))
            b_vocab = pd.read_pickle(os.path.join(args.load_path, "b_vocab_wo.pkl"))
        # """

        pd.to_pickle(vocab, os.path.join(args.load_path, "vocab.pkl"))
        pd.to_pickle(b_vocab, os.path.join(args.load_path, "b_vocab.pkl"))
        if args.direction is None:
            eval_data = MonoTextData(target_pth, True, vocab=vocab, b_vocab=b_vocab)
        elif args.direction:
            eval_data = MonoTextData(
                target_pth.replace(".txt", "1_0.txt"),
                True,
                vocab=vocab,
                b_vocab=b_vocab,
            )
        else:
            eval_data = MonoTextData(
                target_pth.replace(".txt", "0_1.txt"),
                True,
                vocab=vocab,
                b_vocab=b_vocab,
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNNClassifier(len(vocab), 300, [1, 2, 3, 4, 5], 500, 0.5).to(device)
        model.load_state_dict(
            torch.load("checkpoint/%s-classifier.pt" % args.data_name)
        )
        model.eval()
        eval_data, eval_label = eval_data.create_data_batch_label(
            64, device, batch_first=True
        )
        acc = 100 * evaluate(model, eval_data, eval_label)

        print("Acc: %.2f" % acc)

    if args.ac_wo:
        train_pth = os.path.join(data_pth + "_wo", "train_data.txt")
        # """
        if os.path.isfile(
            os.path.join(data_pth + "_wo", "vocab_wo.pkl")
        ) and os.path.isfile(os.path.join(data_pth + "_wo", "b_vocab_wo.pkl")):
            vocab_wo = pd.read_pickle(os.path.join(data_pth + "_wo", "vocab_wo.pkl"))
            b_vocab_wo = pd.read_pickle(
                os.path.join(data_pth + "_wo", "b_vocab_wo.pkl")
            )

        else:
            train_data = MonoTextData(train_pth, True, vocab=100000)
            vocab_wo = train_data.vocab
            b_vocab_wo = train_data.b_vocab
        """
        if os.path.isfile(
            os.path.join(args.load_path, "vocab_wo.pkl")
        ) and os.path.isfile(os.path.join(args.load_path, "b_vocab_wo.pkl")):
            vocab = pd.read_pickle(os.path.join(args.load_path, "vocab_wo.pkl"))
            b_vocab = pd.read_pickle(os.path.join(args.load_path, "b_vocab_wo.pkl"))
        # """

        pd.to_pickle(vocab_wo, os.path.join(data_pth + "_wo", "vocab_wo.pkl"))
        pd.to_pickle(b_vocab_wo, os.path.join(data_pth + "_wo", "b_vocab_wo.pkl"))
        if args.direction is None:
            eval_data = MonoTextData(
                target_pth, True, vocab=vocab_wo, b_vocab=b_vocab_wo
            )
        elif args.direction:
            eval_data = MonoTextData(
                target_pth.replace(".txt", "1_0.txt"),
                True,
                vocab=vocab_wo,
                b_vocab=b_vocab_wo,
            )
        else:
            eval_data = MonoTextData(
                target_pth.replace(".txt", "0_1.txt"),
                True,
                vocab=vocab_wo,
                b_vocab=b_vocab_wo,
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNNClassifier(len(vocab_wo), 300, [1, 2, 3, 4, 5], 500, 0.5).to(device)
        model.load_state_dict(
            torch.load(f"checkpoint/{args.data_name}_wo-classifier.pt")
        )
        model.eval()
        eval_data, eval_label = eval_data.create_data_batch_label(
            64, device, batch_first=True
        )
        acc = 100 * evaluate(model, eval_data, eval_label)

        print("Acc: %.2f" % acc)

    if args.parallel:
        source_pth = os.path.join(data_pth, "test_data_ref.txt")
    elif args.train:
        source_pth = os.path.join(data_pth, "train_data.txt")
    else:
        source_pth = os.path.join(data_pth, "test_data.txt")

    if args.rouge:
        source = pd.read_csv(source_pth, names=["label", "content", "bow"], sep="\t")
        target = pd.read_csv(target_pth, names=["label", "content", "bow"], sep="\t")
        source = source.fillna("").astype({"bow": str}).astype({"content": str})
        target = target.fillna("").astype({"bow": str}).astype({"content": str})
    else:
        source = pd.read_csv(
            source_pth, names=["label", "content"], sep="\t", usecols=[0, 1]
        )
        target = pd.read_csv(
            target_pth, names=["label", "content"], sep="\t", usecols=[0, 1]
        )
        source = source.fillna("").astype({"content": str})
        target = target.fillna("").astype({"content": str})

    sources = []
    targets = []
    count = 0
    if args.trans:
        for i in range(source.shape[0]):
            if args.direction is None:
                if args.rouge:
                    s = source.bow[i].strip()  # .split()
                    t = target.bow[i].strip()  # .split()
                else:
                    s = source.content[i].strip().split()
                    t = target.content[i].strip().split()

                if s == t:
                    count += 1
                if s and t:
                    sources.append([s])
                    targets.append(t)

            elif args.direction and source.label[i]:
                if args.rouge:
                    s = source.bow[i].strip().split()
                    t = target.bow[i].strip().split()
                else:
                    s = source.content[i].strip().split()
                    t = target.content[i].strip().split()
                if s == t:
                    count += 1
                sources.append([s])
                targets.append(t)

            elif not args.direction and not source.label[i]:
                if args.rouge:
                    s = source.bow[i].strip().split()
                    t = target.bow[i].strip().split()
                else:
                    s = source.content[i].strip().split()
                    t = target.content[i].strip().split()
                if s == t:
                    count += 1

                sources.append([s])
                targets.append(t)
    else:
        for i in range(source.shape[0]):
            if args.rouge:
                s = source.bow[i].strip()  # .split()
                t = target.bow[i].strip()  # .split()
            else:
                s = source.content[i].strip().split()
                t = target.content[i].strip().split()
            if s == t:
                count += 1

            if s and t:
                sources.append([s])
                targets.append(t)

    # ROUGE-score
    if args.rouge:
        rouge = Rouge()
        scores = rouge.get_scores(targets, sum(sources, []), avg=True)
        print(f"ROUGE:{scores}")

    print(count, "/", len(targets))

    # BLEU Score
    if args.bleu:
        total_bleu = 0.0
        # print(list(zip(sources, targets))[10000:])

        bleu_values = compute_bleu(sources, targets)
        total_bleu += bleu_values[0]
        total_bleu *= 100
        geo_mean = bleu_values[6] * 100
        print("Bleu: %.2f" % total_bleu)
        print("geo_mean: %.2f" % geo_mean)

    fout_name = "eval"
    if args.ac:
        fout_name += "_ac"
    if args.ac_wo:
        fout_name += "_ac_wo"
    if args.rouge:
        fout_name += "_bow"
    if args.trans:
        fout_name += "_trans"
    if args.direction is not None:
        if args.direction:
            fout_name += "1_0"
        else:
            fout_name += "0_1"
    if args.parallel:
        fout_name += "_para"

    fout_name += ".txt"

    with open(os.path.join(args.load_path, fout_name), "w") as f:
        if args.ac or args.ac_wo:
            f.write("Acc: %.2f\n" % acc)

        if args.bleu:
            f.write("Bleu: %.2f\n" % total_bleu)
            f.write("geo mean: %f\n" % geo_mean)
        if args.rouge:
            f.write("rouge: %s\n" % scores)
        f.write(f"{count}/{len(targets)}")


def add_args(parser):
    parser.add_argument("--data_name", type=str, default="yelp")
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--load_path", type=str)
    parser.add_argument("--trans", default=False, action="store_true")
    parser.add_argument("--train", default=False, action="store_true")
    parser.add_argument("--vocab", default=False, action="store_true")
    parser.add_argument("--b_vocab", default=False, action="store_true")
    parser.add_argument("--ac", default=False, action="store_true")
    parser.add_argument("--ac_wo", default=False, action="store_true")
    parser.add_argument("--bleu", default=False, action="store_true")
    parser.add_argument("--rouge", default=False, action="store_true")
    parser.add_argument("--avg_len", default=False, action="store_true")
    parser.add_argument("--parallel", "-p", default=False, action="store_true")
    parser.add_argument("-d", "--direction", type=int, default=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
