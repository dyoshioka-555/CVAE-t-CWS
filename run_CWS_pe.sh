#!/bin/bash
# セットアップで作成した自分のvirtualenv環境をロード
source ${HOME}/nas02home/mr3venv/bin/activate
#python -u ${HOME}/nas02home/TG/CVAE-t-CWS/evaluate.py --data_name $1 --ac --bleu --load_path "checkpoint/ours-$1-attn/20210909-222926/"
#python -u ${HOME}/nas02home/TG/CVAE-t-CWS/classifier.py --data_name $1
python -u ${HOME}/nas02home/TG/CVAE-t-CWS/run.py --data_name $1 --attn

latest=`ls checkpoint/ours-$1-attn+pe/ -p | tail -n 1`
str="checkpoint/ours-$1-attn+pe/${latest}"
#str="checkpoint/ours-csj_filler-attn+pe/20221013-135111/"

python -u ${HOME}/nas02home/TG/CVAE-t-CWS/reconstract.py --load_path $str --data_name $1 --attn
python -u ${HOME}/nas02home/TG/CVAE-t-CWS/reconstract.py --load_path $str --data_name $1 --attn --trans
#python -u ${HOME}/nas02home/TG/CVAE-t-CWS/reconstract.py --load_path $str --data_name $1 --attn --trans -t "test_text.txt" --out_name "test_trans.txt"
#python -u ${HOME}/nas02home/TG/CVAE-t-CWS/reconstract.py --load_path $str --data_name $1 -v --attn
#python -u ${HOME}/nas02home/TG/CVAE-t-CWS/reconstract.py --load_path $str --data_name $1 -v --attn --trans

python -u ${HOME}/nas02home/TG/CVAE-t-CWS/evaluate.py --load_path $str --data_name $1 --ac --bleu
python -u ${HOME}/nas02home/TG/CVAE-t-CWS/evaluate.py --load_path $str --data_name $1 --ac --bleu --trans

python -u ${HOME}/nas02home/TG/CVAE-t-CWS/recon_to_pos.py --load_path $str --trans -s -d 0
python -u ${HOME}/nas02home/TG/CVAE-t-CWS/recon_to_pos.py --load_path $str --trans -s -d 1
python -u ${HOME}/nas02home/TG/CVAE-t-CWS/evaluate.py --load_path $str --data_name $1 --ac --bleu --trans -d 0
python -u ${HOME}/nas02home/TG/CVAE-t-CWS/evaluate.py --load_path $str --data_name $1 --ac --bleu --trans -d 1

if [ $1 = "csj_filler" ];then
    python -u ${HOME}/nas02home/TG/CVAE-t-CWS/evaluate.py --load_path $str --data_name $1 --bleu --trans -p
elif [ $1 = "csj_filler_latest" ];then
    python -u ${HOME}/nas02home/TG/CVAE-t-CWS/evaluate.py --load_path $str --data_name $1 --bleu --trans -p
fi
if [ $1 = "kansai" ];then
    #python -u ${HOME}/nas02home/TG/CVAE-t-CWS/recon_to_pos.py --load_path $str --trans --wo_bow
    python -u ${HOME}/nas02home/TG/CVAE-t-CWS/evaluate.py --load_path $str --data_name $1 --ac_wo --trans
    python -u ${HOME}/nas02home/TG/CVAE-t-CWS/recon_to_pos.py --load_path $str --trans -s -d 0
    python -u ${HOME}/nas02home/TG/CVAE-t-CWS/recon_to_pos.py --load_path $str --trans -s -d 1
    python -u ${HOME}/nas02home/TG/CVAE-t-CWS/evaluate.py --load_path $str --data_name $1 --ac_wo --trans -d 0
    python -u ${HOME}/nas02home/TG/CVAE-t-CWS/evaluate.py --load_path $str --data_name $1 --ac_wo --trans -d 1
elif [ $1 = "kansai_latest" ];then
    #python -u ${HOME}/nas02home/TG/CVAE-t-CWS/recon_to_pos.py --load_path $str --trans --wo_bow
    python -u ${HOME}/nas02home/TG/CVAE-t-CWS/evaluate.py --load_path $str --data_name $1 --ac_wo --trans
    #python -u ${HOME}/nas02home/TG/CVAE-t-CWS/recon_to_pos.py --load_path $str --trans -s -d 0
    #python -u ${HOME}/nas02home/TG/CVAE-t-CWS/recon_to_pos.py --load_path $str --trans -s -d 1
    python -u ${HOME}/nas02home/TG/CVAE-t-CWS/evaluate.py --load_path $str --data_name $1 --ac_wo --trans -d 0
    python -u ${HOME}/nas02home/TG/CVAE-t-CWS/evaluate.py --load_path $str --data_name $1 --ac_wo --trans -d 1
fi


python -u ${HOME}/nas02home/TG/CVAE-t-CWS/reconstract.py --load_path $str --data_name $1 --attn --trans --dev
python -u ${HOME}/nas02home/TG/CVAE-t-CWS/reconstract.py --load_path $str --data_name $1 --attn --trans --vtrain

###debug###
#python -u ${HOME}/nas02home/TG/CVAE-t-CWS/run2.py --data_name "toda_lecture" --attn

#python preprocess_ja.py --data_name "kansai_wo"
#python -u ${HOME}/nas02home/TG/CVAE-t-CWS/classifier.py --data_name "kansai"
#python -u ${HOME}/nas02home/TG/CVAE-t-CWS/classifier.py --data_name $1