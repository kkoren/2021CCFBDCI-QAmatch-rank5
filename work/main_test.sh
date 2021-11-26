mkdir /home/aistudio/external-libraries
pip install LAC -t /home/aistudio/external-libraries
pip install pypinyin -t /home/aistudio/external-libraries
pip install pyhanlp -t /home/aistudio/external-libraries
pip install ddparser -t /home/aistudio/external-libraries
pip install Levenshtein -t /home/aistudio/external-libraries

cat ./work/raw_data/LCQMC/train ./work/raw_data/BQ/train ./work/raw_data/OPPO/train >./work/raw_data/train.txt

cat ./work/raw_data/LCQMC/dev ./work/raw_data/BQ/dev ./work/raw_data/OPPO/dev >./work/raw_data/dev.txt

python -u -m paddle.distributed.launch --gpus="0" work/code/dev_log.py  \
  --dev_set work/raw_data/dev.txt \
  --device gpu

python -u \
    work/code/predict.py \
    --device gpu \
    --batch_size 128 \
    --input_file ./work/raw_data/test_B_1118.tsv \
    --result_file ./work/user_data/tmp_result/raw_result.csv

python -u ./work/code/result_deal.py