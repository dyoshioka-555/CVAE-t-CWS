import sys
import os
import MeCab
import pandas as pd
import json
import argparse


def pos_pick_mecab(s, selection_pos, wo=False):
    lines = s.split("\n")
    picked_word_list = []  # 「名詞」と「動詞」を格納するリスト
    for line in lines:
        feature = line.split("\t")
        if len(feature) == 2:  # 'EOS'と''を省く
            info = feature[1].split(",")
            if feature[0] != "ー":
                pos = info[0]
                if not wo and pos in selection_pos:
                    picked_word_list.append(feature[0])
                elif wo and pos not in selection_pos:
                    picked_word_list.append(feature[0])
    return " ".join(picked_word_list)


def pos_pick(s, vocab, wo=False):
    words = s.split()
    picked_word_list = []  # 「名詞」と「動詞」を格納するリスト
    for word in words:
        # print(vocab.word2id.get(word, 3))
        # print(vocab.is_cw[vocab.word2id.get(word, 3)])
        if not wo and vocab.is_cw[vocab.word2id.get(word, 3)]:
            picked_word_list.append(word)
        elif wo and not vocab.is_cw[vocab.word2id.get(word, 3)]:
            picked_word_list.append(word)
    return " ".join(picked_word_list)


def main(args):
    if args.input_name:
        file_name = args.input_name
    elif args.target:
        file_name = "test_data.txt"
    elif args.trans:
        file_name = "transfer_results.txt"
    else:
        file_name = "reconstracted_results.txt"
    if args.train:
        file_name = file_name.replace("results.txt", "train_data.txt")
    if args.dev:
        file_name = file_name.replace("results.txt", "dev_data.txt")
    if args.test:
        file_name = file_name.replace("results.txt", "test_data.txt")

    path = os.path.join(args.load_path, file_name)
    if args.output_name:
        out_name = args.output_name
    elif args.target:
        out_name = "bow_tar.txt"
    elif args.bow_only:
        out_name = "trans_bow_only.txt" if args.trans else "recon_bow_only.txt"
    elif args.wo_bow:
        out_name = "trans_wo_bow.txt" if args.trans else "recon_wo_bow.txt"
    elif args.split:
        out_name = "transfer_results.txt" if args.trans else "reconstracted_results.txt"
    else:
        out_name = "trans_bow.txt" if args.trans else "recon_bow.txt"
    if args.train:
        out_name = (
            "transfer_train_data_bow.txt"
            if args.trans
            else "reconstracted_train_data_bow.txt"
        )
    if args.dev:
        out_name = (
            "transfer_dev_data_bow.txt"
            if args.trans
            else "reconstracted_dev_data_bow.txt"
        )
    if args.test:
        out_name = (
            "transfer_test_data_bow.txt"
            if args.trans
            else "reconstracted_test_data_bow.txt"
        )

    if args.direction is not None:
        if args.direction:
            out_name = out_name.replace(".txt", "1_0.txt")
        else:
            out_name = out_name.replace(".txt", "0_1.txt")

    with open(path, "r") as fin, open(
        os.path.join(args.load_path, out_name), "w"
    ) as fout:

        lines = fin.readlines()
        if args.split:
            for line in lines:
                label = int(line.split("\t")[0])
                text = line.split("\t")[1].strip()
                if args.direction and not label:
                    fout.write("%d\t%s\n" % (label, text))
                elif not args.direction and label:
                    fout.write("%d\t%s\n" % (label, text))
        else:
            # vocab = pd.read_pickle(os.path.join("data", args.data_name, "vocab.pkl"))
            # """
            wakati = MeCab.Tagger(
                "-Owakati -u /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd/user_dic.dic"
            )
            parser = MeCab.Tagger(
                "-u /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd/user_dic.dic"
            )
            # "-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd -u /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd/user_dic.dic"
            # """
            for line in lines:
                label = int(line.split("\t")[0])
                text = line.split("\t")[1].strip()
                if args.wo_bow:
                    # """
                    picked_pos = pos_pick_mecab(
                        parser.parse(text), ["名詞", "動詞", "形容詞", "副詞"], True
                    ).strip()
                    # """
                    # picked_pos = pos_pick(text, vocab, True).strip()
                    if args.direction is not None:
                        if args.direction and not label:
                            fout.write("%d\t%s\n" % (label, picked_pos))
                        elif not args.direction and label:
                            fout.write("%d\t%s\n" % (label, picked_pos))
                    else:
                        fout.write("%d\t%s\n" % (label, picked_pos))
                else:
                    # """
                    picked_pos = pos_pick_mecab(
                        parser.parse(text), ["名詞", "動詞", "形容詞", "副詞"]
                    ).strip()
                    # """
                    # picked_pos = pos_pick(text, vocab).strip()
                    if args.bow_only or args.target:
                        fout.write("%s\n" % (picked_pos))
                    elif args.direction is not None:
                        if args.direction and not label:
                            if args.parse_text:
                                fout.write(
                                    "%d\t%s\t%s\n"
                                    % (
                                        label,
                                        wakati.parse(line.split("\t")[1]).strip(),
                                        picked_pos,
                                    )
                                )
                            else:
                                fout.write(
                                    "%d\t%s\t%s\n"
                                    % (label, line.split("\t")[1].strip(), picked_pos)
                                )
                        elif not args.direction and label:
                            if args.parse_text:
                                fout.write(
                                    "%d\t%s\t%s\n"
                                    % (
                                        label,
                                        wakati.parse(line.split("\t")[1]).strip(),
                                        picked_pos,
                                    )
                                )
                            else:
                                fout.write(
                                    "%d\t%s\t%s\n"
                                    % (label, line.split("\t")[1].strip(), picked_pos)
                                )
                    else:
                        if args.parse_text:
                            fout.write(
                                "%d\t%s\t%s\n"
                                % (
                                    label,
                                    wakati.parse(line.split("\t")[1]).strip(),
                                    picked_pos,
                                )
                            )
                        else:
                            fout.write(
                                "%d\t%s\t%s\n"
                                % (label, line.split("\t")[1].strip(), picked_pos)
                            )


def add_args(parser):
    parser.add_argument("--parser", type=str, default="mecab", help="parser name")
    parser.add_argument(
        "--parse_text",
        default=False,
        action="store_true",
        help="parse original text",
    )
    parser.add_argument("--input_name", "-i", type=str, default=None, help="input file name")
    parser.add_argument("--output_name", "-o", type=str, default=None, help="input file name")
    parser.add_argument("--load_path", type=str)
    parser.add_argument(
        "--data_name",
        type=str,
        default=None,
    )
    parser.add_argument("--trans", default=False, action="store_true")
    parser.add_argument("--train", default=False, action="store_true")
    parser.add_argument("--dev", default=False, action="store_true")
    parser.add_argument("--test", default=False, action="store_true")
    parser.add_argument("--bow_only", "-b", default=False, action="store_true")
    parser.add_argument("--wo_bow", "-w", default=False, action="store_true")
    parser.add_argument("--target", "-t", default=False, action="store_true")
    parser.add_argument("--split", "-s", default=False, action="store_true")
    parser.add_argument("-d", "--direction", type=int, default=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
