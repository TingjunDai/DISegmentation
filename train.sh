# Example: `./train.sh BiRefNet 0,1,2,3,4,5,6`

method=${1:-"BiRefNet"}
case "${method}" in
    'BiRefNet') epochs=100 ;;
    'ISNet') epochs=10000 ;;
    'ISNet_GTEncoder') epochs=10000;;
    'UDUN') epochs=48 ;;
    'MVANet') epochs=80 ;;
esac

devices=$2
nproc_per_node=$(echo ${devices} | grep -o "," | wc -l) 
nproc_per_node=$((nproc_per_node + 1))

to_be_distributed=$(echo ${nproc_per_node} | awk '{if($1 > 1) print "True"; else print "False";}')

echo Training started at $(date)

if [ ${to_be_distributed} == "True" ]; then
    echo "Multi-GPU mode received..."
    CUDA_VISIBLE_DEVICES=${devices} \
    setsid nohup mpirun -np ${nproc_per_node} --allow-run-as-root \
    python train.py --ckpt_dir ckpt/${method} --epochs ${epochs} \
        > nohup.log 2>&1 &
else
    echo "Single-GPU mode received..."
    CUDA_VISIBLE_DEVICES=${devices} \
    setsid nohup python train.py --ckpt_dir ckpt/${method} --epochs ${epochs} \
        > nohup.log 2>&1 &
fi

echo Training finished at $(date)