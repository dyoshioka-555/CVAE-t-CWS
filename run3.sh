#!/bin/bash
# セットアップで作成した自分のvirtualenv環境をロード
source ${HOME}/nas02/home/mr3venv/bin/activate
# 作成したPythonスクリプトを実行
#python -u ${HOME}/nas02home/TG/CVAE-t-CWS/run_c4.py --data_name $1 --attn
declare -i op_vis=0
declare -i op_rouge=0
declare -i op_wer=0
declare -i op_wo_bow=0
declare -i op_cycle=0
declare -i op_cycle_cwape=0
declare -i op_cycle_mix_cwape=0
declare -i op_cycle_ft_cwape=0
declare -i op_cycle_few_cwape=0
declare -i op_cycle_para_cwape=0
declare -i op_cycle2=0
declare -i op_cycle2_mix=0
declare -i op_bow=0
declare -i op_attn=0
declare -i op_attn_pe=0
declare -i op_attn_cwape=0


#op_vis=1; echo "vis"
#op_rouge=1; echo "rouge"
op_wer=1; echo "wer"
#op_wo_bow=1; echo "wo_bow"

#op_cycle=1; echo "cycle"
if [ $2 = "cwape" ]; then
    op_cycle_cwape=1; echo "cycle-cwape"
elif [ $2 = "cycle" ]; then
    op_cycle=1; echo "cycle"
elif [ $2 = "mix" ]; then
    op_cycle_mix_cwape=1; echo "cycle_mix-cwape"
elif [ $2 = "few" ]; then
    op_cycle_few_cwape=1; echo "cycle_few-cwape"
elif [ $2 = "ft" ]; then
    op_cycle_ft_cwape=1; echo "cycle_ft-cwape"
elif [ $2 = "para" ]; then
    op_cycle_para_cwape=1; echo "cycle_para-cwape"
elif [ $2 = "2" ]; then
    op_cycle2=1; echo "cycle2"
elif [ $2 = "2mix" ]; then
    op_cycle2_mix=1; echo "cycle2_mix"
elif [ $2 = "bow" ]; then
    op_bow=1; echo "bow"
elif [ $2 = "attn" ]; then
    op_attn=1; echo "attn"
elif [ $2 = "attn_pe" ]; then
    op_attn_pe=1; echo "attn_pe"
elif [ $2 = "attn_cwape" ]; then
    op_attn_cwape=1; echo "attn_cwape"
fi


if [ $op_cycle -eq 1 ];then
    latest=`ls checkpoint/ours-$1-cycle/ -p | tail -n 2 |head -n 1`
    str="checkpoint/ours-$1-cycle/${latest}"
    #str="checkpoint/ours-$1-cycle/20220819-131938/"
 
elif [ $op_cycle_cwape -eq 1 ];then
    latest=`ls checkpoint/ours-$1-cycle-cwape/ -p | tail -n 2 |head -n 1`
    str="checkpoint/ours-$1-cycle-cwape/${latest}"
    #str="checkpoint/ours-$1-cycle-cwape/20221116-150125/"

elif [ $op_cycle_mix_cwape -eq 1 ];then

    latest=`ls checkpoint/ours-$1-cycle_mix_$3-cwape/ -p | tail -n 2 |head -n 1`
    str="checkpoint/ours-$1-cycle_mix_$3-cwape/${latest}"
    #str="checkpoint/ours-$1-cycle_mix_5000-cwape/20221104-163506/"

elif [ $op_cycle_few_cwape -eq 1 ];then

    latest=`ls checkpoint/ours-$1-cycle_few_$3-cwape/ -p | tail -n 2 |head -n 1`
    str="checkpoint/ours-$1-cycle_few_$3-cwape/${latest}"
    #str="checkpoint/ours-$1-cycle_mix-cwape/20220819-131938/"
    
elif [ $op_cycle_ft_cwape -eq 1 ];then

    latest=`ls checkpoint/ours-$1-cycle_ft_$3-cwape/ -p | tail -n 2 |head -n 1`
    str="checkpoint/ours-$1-cycle_ft_$3-cwape/${latest}"
    #str="checkpoint/ours-$1-cycle_ft_5000-cwape/20221107-162835/"

elif [ $op_cycle_para_cwape -eq 1 ];then

    latest=`ls checkpoint/ours-$1-cycle_para-cwape/ -p | tail -n 2 |head -n 1`
    str="checkpoint/ours-$1-cycle_para-cwape/${latest}"

elif [ $op_cycle2 -eq 1 ];then

    latest=`ls checkpoint/ours-$1-cycle2/ -p | tail -n 2 |head -n 1`
    str="checkpoint/ours-$1-cycle2/${latest}"
    #str="checkpoint/ours-$1-cycle2/20220819-131938/"

elif [ $op_cycle2_mix -eq 1 ];then

    latest=`ls checkpoint/ours-$1-cycle2_mix_$3/ -p | tail -n 2 |head -n 1`
    str="checkpoint/ours-$1-cycle2_mix_$3/${latest}"
    #str="checkpoint/ours-$1-cycle2_mix/20220819-131938/"

elif [ $op_bow -eq 1 ];then

    latest=`ls checkpoint/ours-$1-bow/ -p | tail -n 2 |head -n 1`
    str="checkpoint/ours-$1-bow/${latest}"

elif [ $op_attn -eq 1 ];then

    latest=`ls checkpoint/ours-$1-attn/ -p | tail -n 2 |head -n 1`
    str="checkpoint/ours-$1-attn/${latest}"
    str="checkpoint/ours-csj_filler_latest-attn/20230131-152844/"

elif [ $op_attn_pe -eq 1 ];then

    latest=`ls checkpoint/ours-$1-attn+pe/ -p | tail -n 2 |head -n 1`
    str="checkpoint/ours-$1-attn+pe/${latest}"
    #str="checkpoint/ours-csj_filler_latest-attn+pe/20221116-151900/"

elif [ $op_attn_cwape -eq 1 ];then

    latest=`ls checkpoint/ours-$1-attn+cwape/ -p | tail -n 2 |head -n 1`
    str="checkpoint/ours-$1-attn+cwape/${latest}"
else 

    latest=`ls checkpoint/ours-$1/ -p | tail -n 2 |head -n 1`
    str="checkpoint/ours-$1/${latest}"
fi


if [ $op_vis -eq 1 ]; then
python -u ${HOME}/nas02/home/TG/CVAE-t-CWS/attn_visualize.py --load_path $str --data_name $1
python -u ${HOME}/nas02/home/TG/CVAE-t-CWS/attn_visualize.py --load_path $str --data_name $1 --trans
#python -u ${HOME}/nas02/home/TG/CVAE-t-CWS/attn_visualize.py --load_path $str --data_name $1 --train
#python -u ${HOME}/nas02/home/TG/CVAE-t-CWS/attn_visualize.py --load_path $str --data_name $1 --train  --trans
fi

if [ $op_rouge -eq 1 ]; then
    python -u ${HOME}/nas02/home/TG/CVAE-t-CWS/recon_to_pos.py --load_path $str 
    python -u ${HOME}/nas02/home/TG/CVAE-t-CWS/evaluate.py --load_path $str --data_name $1 --rouge
    python -u ${HOME}/nas02/home/TG/CVAE-t-CWS/recon_to_pos.py --load_path $str  --trans 
    python -u ${HOME}/nas02/home/TG/CVAE-t-CWS/evaluate.py --load_path $str --data_name $1 --rouge  --trans
fi

if [ $op_wer -eq 1 ]; then
    python -u ${HOME}/nas02/home/TG/CVAE-t-CWS/recon_to_pos.py --data_name $1 --load_path "data/$1" --target --bow_only
    python -u ${HOME}/nas02/home/TG/CVAE-t-CWS/recon_to_pos.py --data_name $1 --load_path $str --bow_only
    python -u wer.py "data/$1/bow_tar.txt" "${str}recon_bow_only.txt" > "${str}wer_recon.txt"
    python -u ${HOME}/nas02/home/TG/CVAE-t-CWS/recon_to_pos.py --data_name $1 --load_path $str --trans --bow_only
    python -u wer.py "data/$1/bow_tar.txt" "${str}trans_bow_only.txt" > "${str}wer.txt"

fi

if [ $op_wo_bow -eq 1 ]; then
    python -u ${HOME}/nas02/home/TG/CVAE-t-CWS/recon_to_pos.py --load_path $str --trans --wo_bow
    python -u ${HOME}/nas02/home/TG/CVAE-t-CWS/recon_to_pos.py --load_path $str --trans --wo_bow -d 0
    python -u ${HOME}/nas02/home/TG/CVAE-t-CWS/recon_to_pos.py --load_path $str --trans --wo_bow -d 1

fi

#python -u ${HOME}/nas02/home/TG/CVAE-t-CWS/recon_to_pos.py --load_path "checkpoint/ours-$1-attn/20220511-112132/" --trans --wo_bow
#python -u ${HOME}/nas02/home/TG/CVAE-t-CWS/recon_to_pos.py --load_path "checkpoint/ours-$1-attn/20220511-112132/" --trans --wo_bow -d 0
#python -u ${HOME}/nas02/home/TG/CVAE-t-CWS/recon_to_pos.py --load_path "checkpoint/ours-$1-attn/20220511-112132/" --trans --wo_bow -d 1
#python -u ${HOME}/nas02/home/TG/CVAE-t-CWS/recon_to_pos.py --load_path "checkpoint/ours-kansai-attn/20220511-112131" --trans --wo_bow
#python -u ${HOME}/nas02/home/TG/CVAE-t-CWS/recon_to_pos.py --load_path "checkpoint/ours-kansai-attn/20220511-112131" --trans --wo_bow -d 0
#python -u ${HOME}/nas02/home/TG/CVAE-t-CWS/recon_to_pos.py --load_path "checkpoint/ours-kansai-attn/20220511-112131" --trans --wo_bow -d 1
