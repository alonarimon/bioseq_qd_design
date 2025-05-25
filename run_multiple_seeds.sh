
export CUDA_VISIBLE_DEVICES=2 # Set to the GPU you want to use - GPU2 (3060)

# List of random seeds
seeds=(0 1 2)

# Run the script for each seed
for seed in "${seeds[@]}"
do
    echo "Running with seed: $seed, mutation_model: mutator_helix, fitness_model: fitness_utr_ensemble, mutation_length: 1"
    python run_elm.py qd.seed=$seed env.seed=$seed mutation_model=mutator_helix fitness_model=fitness_utr_ensemble mutation_model.mutation_length=1
    
    echo "Running with seed: $seed, mutation_model: random, fitness_model: fitness_utr_ensemble, mutation_length: 1"
    python run_elm.py qd.seed=$seed env.seed=$seed mutation_model=random fitness_model=fitness_utr_ensemble mutation_model.mutation_length=1
    
    echo "Running with seed: $seed, mutation_model: mutator_helix, fitness_model: fitness_helix_mrna, mutation_length: 1"
    python run_elm.py qd.seed=$seed env.seed=$seed mutation_model=mutator_helix fitness_model=fitness_helix_mrna mutation_model.mutation_length=1
    
    echo "Running with seed: $seed, mutation_model: random, fitness_model: fitness_helix_mrna, mutation_length: 1"
    python run_elm.py qd.seed=$seed env.seed=$seed mutation_model=random fitness_model=fitness_helix_mrna mutation_model.mutation_length=1
done