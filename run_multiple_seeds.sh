
export CUDA_VISIBLE_DEVICES=2 # Set to the GPU you want to use - GPU2 (3060)

# List of random seeds
seeds=(0 1 2)
config_dir="conf"
config_name="utr_task_similarity_based"
total_steps=35000
save_snapshot_interval=100
wandb_group="run_elm"
visualize_on_interval=True
init_from_offline_data=False

# Run the script for each seed
for seed in "${seeds[@]}"
do
    
    echo "Running with seed: $seed, mutation_model: random, fitness_model: fitness_helix_mrna"
    python run_elm.py --config-dir=$config_dir --config-name=$config_name \
    qd.seed=$seed env.seed=$seed \
    mutation_model=random fitness_model=fitness_helix_mrna \
    qd.save_snapshot_interval=$save_snapshot_interval qd.total_steps=$total_steps qd.init_from_offline_data=$init_from_offline_data \
    wandb_group=$wandb_group qd.visualize_on_interval=$visualize_on_interval \
    
    echo "Running with seed: $seed, mutation_model: mutator_helix, fitness_model: fitness_utr_ensemble"
    python run_elm.py --config-dir=$config_dir --config-name=$config_name \
    qd.seed=$seed env.seed=$seed \
    mutation_model=mutator_helix fitness_model=fitness_utr_ensemble \
    qd.save_snapshot_interval=$save_snapshot_interval qd.total_steps=$total_steps  qd.init_from_offline_data=$init_from_offline_data \
    wandb_group=$wandb_group qd.visualize_on_interval=$visualize_on_interval \

    echo "Running with seed: $seed, mutation_model: random, fitness_model: fitness_utr_ensemble"
    python run_elm.py --config-dir=$config_dir --config-name=$config_name \
    qd.seed=$seed env.seed=$seed \
    mutation_model=random fitness_model=fitness_utr_ensemble \
    qd.save_snapshot_interval=$save_snapshot_interval qd.total_steps=$total_steps  qd.init_from_offline_data=$init_from_offline_data \
    wandb_group=$wandb_group qd.visualize_on_interval=$visualize_on_interval \

    echo "Running with seed: $seed, mutation_model: mutator_helix, fitness_model: fitness_helix_mrna"
    python run_elm.py --config-dir=$config_dir --config-name=$config_name \
    qd.seed=$seed env.seed=$seed \
    mutation_model=mutator_helix fitness_model=fitness_helix_mrna \
    qd.save_snapshot_interval=$save_snapshot_interval qd.total_steps=$total_steps  qd.init_from_offline_data=$init_from_offline_data \
    wandb_group=$wandb_group qd.visualize_on_interval=$visualize_on_interval \
    
done