#!/bin/bash
#SBATCH --job-name=grid_pv_hpo
#SBATCH --output=logs/hpo_%j.out
#SBATCH --error=logs/hpo_%j.err
#SBATCH --nodes=7
#SBATCH --gres=gpu:1  
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1  # 1 task (Ray process) per node
#SBATCH --time=72:00:00
#SBATCH -p cpu-gpu-v100

# --- Setup ---
source ~/.bashrc
mkdir -p logs

# --- Ray Network Setup ---
# 1. Get the Head Node IP
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# 2. Port Configuration
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "Head Node IP: $head_node_ip"
echo "Ray Head Address: $ip_head"

# 3. Start Ray Head Node
echo "Starting Ray Head on $head_node"
# We use & to run in background so shell can proceed to start workers
# Using srun -w to explicitly target the head node
srun --nodes=1 --ntasks=1 -w "$head_node" \
    uv run ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus=1 \
    --block &

# 4. Start Ray Worker Nodes
# We sleep briefly to let the head node initialize
sleep 20
worker_num=$((SLURM_JOB_NUM_NODES - 1))
if [ "$worker_num" -gt 0 ]; then
    echo "Starting $worker_num Ray Workers"
    # srun will automatically distribute these across the remaining nodes
    # (or you can be explicit with -w, but usually --nodes=1 etc is correct context)
    # The trick with srun is ensuring it doesn't try to re-run on head node if using overlapping settings
    # Best practice: srun checks existing allocation.
    # We execute on all nodes EXCEPT head node
    # Getting worker list:
    # A cleaner way in Slurm is just running 'ray start' on ALL nodes, but passing address to all.
    # The one that is head (already started) will fail or we can use specific login.
    # Standard snippet approach:
    
    # Iterate through remaining nodes
    for ((i=1; i<${#nodes_array[@]}; i++)); do
        node_i=${nodes_array[$i]}
        echo "Starting Worker on $node_i"
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            uv run ray start --address "$ip_head" \
            --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus=1 \
            --block &
    done
fi

# 5. Wait for cluster to be ready
sleep 15

# --- Run the Job ---
# Set the address so the python script picks it up (ray.init('auto'))
# export RAY_ADDRESS="http://127.0.0.1:8265"  # Usually ray client, but for script in cluster:
# Actually, for running the driver on the head node:
export RAY_ADDRESS=$ip_head

export RAY_RUNTIME_ENV_WORKING_DIR_UPLOAD_DISABLED=1

echo "Submitting training job to Ray Cluster..."
uv run python -u /mnt/home/osanghi/pvcracks/pvcracks/MultiSolSegment/training/train_channeled_unet_cv_hpo.py

# --- Cleanup ---
# Not strictly necessary as Slurm kills the job, but good practice
# uv run ray stop