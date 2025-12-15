#Please find all pretrain result in ./outputs/*
time=$(date "+%Y_%m_%d_%H:%M:%S")
if [ ! -d "./outputs" ]; then
       mkdir outputs
fi
pretrain_file_name="pretrain_log"
nohup mpirun -n 2 \
                  python run_pretrain.py \
                         --device_target Ascend \
                         --amp True \
                         --jit True \
                         --lr 2e-5 \
                         --warmup_steps 10000 \
                         --train_batch_size 256 \
                         --epochs 15 \
                         --save_steps 10000 \
                         --do_load_ckpt False \
                         --config config/bert_config_small.json \
                         >> ./outputs/${pretrain_file_name}_${time}.log 2>&1 &