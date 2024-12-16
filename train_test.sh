# Example: `setsid nohup ./train_test.sh BiRefNet 0,1,2,3,4,5,6 0 &>nohup.log &`

method=${1:-"BiRefNet"}
devices=${2:-0}

bash train.sh ${method} ${devices}

devices_test=${3:-0}
pred_root=pred_${method} # inference result save path
eval_root=eval_${method} # evaluation result save path

bash test.sh ${devices_test} ${pred_root} ${eval_root} ${method}

hostname