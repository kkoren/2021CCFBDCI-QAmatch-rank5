mkdir /home/aistudio/external-libraries
pip install LAC -t /home/aistudio/external-libraries
pip install pypinyin -t /home/aistudio/external-libraries
pip install pyhanlp -t /home/aistudio/external-libraries
pip install ddparser -t /home/aistudio/external-libraries
pip install Levenshtein -t /home/aistudio/external-libraries

cat ./work/raw_data/LCQMC/train ./work/raw_data/BQ/train ./work/raw_data/OPPO/train >./work/raw_data/train.txt

cat ./work/raw_data/LCQMC/dev ./work/raw_data/BQ/dev ./work/raw_data/OPPO/dev >./work/raw_data/dev.txt

python -u ./work/code/eda.py

python -u -m paddle.distributed.launch --gpus "0" work/code/train.py \
       --train_set ./work/user_data/eda_data/gaiic_train_eda.txt \
       --dev_set ./work/raw_data/dev.txt \
       --device gpu \
       --eval_step 100 \
       --save_dir ./work/user_data/checkpoints \
       --train_batch_size 128 \
       --learning_rate 3E-5 \
       --rdrop_coef 0.0 \
       --seed 2021 \
       --weight_decay 1E-3 \
       --max_seq_length 62 \
       --warmup_proportion 0.1


python -u \
    work/code/predict.py \
    --device gpu \
    --batch_size 128 \
    --input_file ./work/raw_data/test_B_1118.tsv \
    --result_file ./work/work/user_data/tmp_result/raw_result.csv

python -u ./work/code/result_deal.py