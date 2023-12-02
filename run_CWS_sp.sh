#!/bin/bash
# セットアップで作成した自分のvirtualenv環境をロード
source ${HOME}/nas02home/mr3venv/bin/activate
#python -u ${HOME}/nas02home/TG/CVAE-t-CWS/evaluate.py --data_name $1 --ac --bleu --load_path "checkpoint/ours-$1-attn/20210909-222926/"
#python -u ${HOME}/nas02home/TG/CVAE-t-CWS/classifier.py --data_name $1
python -u ${HOME}/nas02home/TG/CVAE-t-CWS/run_sp.py --data_name $1 
