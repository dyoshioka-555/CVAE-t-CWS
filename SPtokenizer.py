
import argparse
import os
import sentencepiece as spm
import pandas as pd

def main(args):
	# データファイルのパスを指定
	data_path = os.path.join("data", args.data_name)
	data_file = os.path.join(data_path, 'train_data.txt')

	# データを読み込む（各行が タブ区切りで ラベル、テキスト、内容語 になっていると仮定）
	df = pd.read_csv(data_file, sep='\t', header=None)
	df.columns = ['label', 'text', 'content_words']

	# テキスト部分のみを抽出し、一時ファイルに保存
	text_file = 'texts.txt'
	df['text'].to_csv(text_file, index=False, header=False)

	if args.train:
		# SentencePieceのモデルをトレーニング
		spm.SentencePieceTrainer.train(input=text_file, model_prefix=args.data_name, vocab_size=args.size, character_coverage=1, pad_id=3, pad_piece='<pad>')
		# 生成されたモデルファイルは 'mymodel.model' と 'mymodel.vocab' になります
	
	elif args.test:
		sp = spm.SentencePieceProcessor()
		sp.Load(f"{args.data_name}.model")
		# テストデータのパスを正しく設定
		test_data_file = os.path.join(data_path, 'test_data.txt')

		# テストデータを読み込む
		df = pd.read_csv(test_data_file, sep='\t', header=None)
		df.columns = ['label', 'text', 'content_words']

		# 未知語（unknown words）のカウント
		unknown_word_count = 0

		# 各テストデータに対してトークン化を行い、未知語をカウント
		for line in df['text']:
			ids = sp.EncodeAsIds(line.strip())
			unknown_word_count += sum(1 for id in ids if id == 0)

		# 未知語の数を出力
		print(unknown_word_count)


		unknown_word_count = 0
		vocabulary_file = data_path + f"/vocab_{args.data_name}.txt"
		
		# 語彙リストを読み込み、セットに変換
		with open(vocabulary_file, 'r', encoding='utf-8') as file:
			vocabulary = set(file.read().splitlines())

		# テストデータを単語に分割（空白で分割されていると仮定）
		for line in df['text']:
			words = line.strip().split(" ")
			unknown_word_count += sum(1 for word in words if word not in vocabulary)

		# 結果の出力
		print(unknown_word_count)
	
	else:
		sp = spm.SentencePieceProcessor()
		sp.Load(f"{args.data_name}.model")
		print(sp.GetPieceSize())

		text = df['text'][65]
		proto = sp.encode(text, out_type='immutable_proto')
		print(proto)
		print(proto.surface)
		print(proto.id)

		ids = sp.EncodeAsIds(text)
		print(ids)
		print(sp.DecodeIds(ids))
		
		ids = [435, 5, 85, 30404, 11073, 9, 7479, 11]
		print(sp.DecodeIds(ids))

		text = "ＰＳＰ は 大型 液晶 画面 と 光学 ドライブ 高性能 マイクロプロセッサ を 搭載 し た"
		tokens = sp.EncodeAsPieces(text)
		print(tokens)
		print(sp.DecodePieces(tokens))

		ids = sp.EncodeAsIds(text)
		print(ids)
		print(sp.DecodeIds(ids))
	

def add_args(parser):
    parser.add_argument(
        "--data_name", type=str, default="csj_fillert_latest", help="data name"
    )
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--train", default=False, action="store_true")
    parser.add_argument("--test", default=False, action="store_true")
    parser.add_argument("--size", type=int, default=32000)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)