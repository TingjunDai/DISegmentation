devices=${1:-0}
pred_root=${2:-e_mvanet}
eval_root=${3:-e_res_mvanet}
method=${4:-"BiRefNet"}

# Inference

CUDA_VISIBLE_DEVICES=${devices} python inference.py --pred_root ${pred_root} --ckpt_folder ckpt/${method}

echo Inference finished at $(date)

# Evaluation
log_dir=e_logs_${method} && mkdir ${log_dir}

task=$(python3 config.py --print_task)
testsets=$(python3 config.py --print_testsets)

testsets=(`echo ${testsets} | tr ',' ' '`) && testsets=${testsets[@]}

for testset in ${testsets}; do
    # python eval_existingOnes.py --pred_root ${pred_root} --data_lst ${testset} > ${log_dir}/eval_${testset}.out
    nohup python eval.py --pred_root ${pred_root} --data_lst ${testset} --save_dir ${eval_root} > ${log_dir}/eval_${testset}.out 2>&1 &
done


echo Evaluation started at $(date)
