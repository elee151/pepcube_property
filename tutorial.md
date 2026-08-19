# Pepcube Tutorial

Examples on how to run the various scripts for pepcube_transformer (ChemBERTa, PepDoRA) and pepcube_chemprop (classical regression models, chemprop, chemeleon)

## Setup

All scripts rely on the base configuration and variables set in `config.py` such as the pipeline mode (`chemprop`, `chemeleon`, `pretrained`, etc.).
The following have been used to submit the jobs or are examples on how to submit.

- To customize a run, overwrite the relevant environment variables when launching a script (examples below).
- To add or change datasets/configurations, edit the config file directly.
- Full_run in the configuration represents the finalized multihead datasets
- For models that load a pretrained checkpoint, you can eitherspecify a directory in (--pretrain_ckpt) or let the script resolve one from a path built off your inputs. This path structure is specific (see `config.py`) so if your checkpoint won't match it, just pass the path explicitly via model_dir
- Most scripts logs a standard output that incorporate chemprop and the transformers output along with extras to track progress. General results are saved as jsons and the results.py script writes them to one of three results files

## Transformer Submissions 
### conda env  = pepcube_transformer

### LoRA fine-tuning

```bash
ALL_SPLITS="random group"

nohup bash -c '
for split in '"$ALL_SPLITS"'; do
MODEL_ARCH=multi_head PIPELINE_MODE=lora_finetune TORCH_FORCE_WEIGHTS_ONLY_LOAD=0 ACCELERATOR=gpu SPLIT_STRATEGY="$split" NUM_WORKERS=4 HF_HOME=/home/rbirolo/pepcube_property/hf_cache FINETUNE_EPOCHS=30 FINETUNE_BATCH=16 FINETUNE_LR=2e-5 WARMUP_FRAC=0.15 WEIGHT_DECAY=0.01 MAX_GRAD_NORM=1.0 MAX_LENGTH=512 POOLING=mean LORA_R=8 LORA_ALPHA=32 LORA_DROPOUT=0.1 nohup python transformer/lora_chemberta_multitask.py --run_id full_run --pretrain_ckpt /home/rbirolo/pepcube_property/lora_chemberta/backbone --lora_targets query,key,value
done
'> ./logs/$(date +%Y-%m-%d)_lora_finetune_full_run_both_splits.log 2>&1 &
```

### Fine-tuning & evaluation (default ChemBERTa or PepDoRA)

```bash
ALL_MODELS="chemberta-77m-mtr pepdora"
ALL_SPLITS="random group"

nohup bash -c '
for model in '"$ALL_MODELS"'; do
   for split in '"$ALL_SPLITS"'; do
    MODEL_ARCH=multi_head PIPELINE_MODE=transformer TORCH_FORCE_WEIGHTS_ONLY_LOAD=0 BASE_DIR=/home/rbirolo/pepcube_property SPLIT_STRATEGY="$split" NUM_WORKERS=4 HF_HOME=/home/rbirolo/pepcube_property/hf_cache HF_HUB_CACHE=/home/rbirolo/pepcube_property/hf_cache/hub nohup python transformer/finetune_transformer_multitask.py --model_name "$model" --run_id full_run --lr 2e-5 --batch_size 16 --epochs 30 --dropout 0.1 --freeze_encoder_epochs 0 --warmup_steps 100 
  done
done
'> ./logs/$(date +%Y-%m-%d)_finetune_transformer_defaults_full_default.log 2>&1 &
```

### Hyperparameter optimization

```bash
nohup python transformer/hpo_chemberta.py --subset_name all_generated_scored_hpo100k_chemberta > ./logs/$(date +%Y-%m-%d)_hpo_chemberta_full.log 2>&1 &
```

### Prediction

LoRA-based:

```bash
nohup bash -c '
DATASETS=(
"./data/mutated_experimental_peptides_90k.csv"
"./data/converted_literature_sequences.csv"
"./data/experimental_sequences.csv"
)
for input_csv in "${DATASETS[@]}"; do
tag=$(basename "$input_csv" .csv)
SPLIT_STRATEGY=random ACCELERATOR=gpu MODEL_ARCH=multi_head PIPELINE_MODE=lora_finetune TORCH_FORCE_WEIGHTS_ONLY_LOAD=0 NUM_WORKERS=4 HF_HOME=/home/rbirolo/pepcube_property/hf_cache FINETUNE_EPOCHS=30 FINETUNE_BATCH=16 FINETUNE_LR=2e-5 WARMUP_FRAC=0.15 WEIGHT_DECAY=0.01 MAX_GRAD_NORM=1.0 MAX_LENGTH=512 POOLING=mean LORA_R=8 LORA_ALPHA=32 LORA_DROPOUT=0.1 nohup python transformer/predict_lora_chemberta.py --model_dir /home/rbirolo/pepcube_property/runs/chemberta_lora_random_raw/finetune/run3_new_chemberta_synth_camsol10k_hemo550/final_model --input_csv "$input_csv" --output_csv "predictions/${tag}_lora_90_trainval.csv"
done
'> ./logs/$(date +%Y-%m-%d)lora_predictions_90_trainval.log 2>&1 &
echo "PID: $!"
```

Default transformer:

```bash
nohup bash -c '
DATASETS=(
"./data/mutated_experimental_peptides_90k.csv"
"./data/converted_literature_sequences.csv"
"./data/experimental_sequences.csv"
)
for input_csv in "${DATASETS[@]}"; do
tag=$(basename "$input_csv" .csv)
SPLIT_STRATEGY=random TORCH_FORCE_WEIGHTS_ONLY_LOAD=0 BASE_DIR=/home/rbirolo/pepcube_property SPLIT_STRATEGY="$split" NUM_WORKERS=4 HF_HOME=/home/rbirolo/pepcube_property/hf_cache HF_HUB_CACHE=/home/rbirolo/pepcube_property/hf_cache/hub nohup python transformer/predict_transformer_multitask.py --model_dir runs/finetune_transformer/chemberta-77m-mtr/full_run/final_model --model_name chemberta-77m-mtr --input_csv "$input_csv" --output_csv "predictions/${tag}_chemberta_90_trainval.csv"
done
'> ./logs/$(date +%Y-%m-%d)_chemberta_def_predictions_90_trainval.log 2>&1 &
echo "PID: $!"
```
### Single-task Transformer (ChemBERTa, PepDoRA)

```bash
ALL_MODES="chemberta-77m-mtr pepdora"
ALL_DATASETS="synthesizability chemberta_hemolysis_chemprop_550 even_chemberta_noncanonical_camsol_chemprop_10k"
ALL_SPLITS="random group"
nohup bash -c '
for mode in '"$ALL_MODES"'; do
  for dataset in '"$ALL_DATASETS"'; do
    for split in '"$ALL_SPLITS"'; do
      echo "===Mode=$mode dataset=$dataset split=$split==="
      SPLIT_STRATEGY="$split" ACCELERATOR=gpu python transformer/finetune_transformer_singletask.py --model_name "$mode" --dataset_name "$dataset"
    done
  done
done
' > ./logs/$(date +%Y-%m-%d)_finetune_singletask_transformer_all.log 2>&1 &
echo "PID: $!"
```
### Pretraining

```bash
ACCELERATOR=gpu SPLIT_STRATEGY=group NUM_WORKERS=4 HF_HOME=/home/rbirolo/pepcube_property/hf_cache CHEMBERTA_MODEL=/home/rbirolo/pepcube_property/hf_cache/hub/models--DeepChem--ChemBERTa-77M-MTR/snapshots/fc007d31c2fb774ab7a8e5a8d318e25cb01d2da1 nohup python transformer/pretrain_chemberta.py --subset_name new_1M --cluster_col cluster_umap_hdb > ./logs/$(date +%Y-%m-%d)_pretrain_chemberta_trial.log
```

## ChemProp Submissions 
### conda env  = pepcube_chemprop

### Finetuning Multitask Pretrained Model - using optimized hyperparameters from HPO 

```bash
ALL_SPLITS="random group"
nohup bash -c '
for split in '"$ALL_SPLITS"'; do
  ACCELERATOR=gpu MODEL_ARCH=multi_head MPNN_DEPTH=6 MPNN_D_H=600 FINETUNE_EPOCHS=20 FINETUNE_BATCH=64 FINETUNE_LR=0.0005 DROPOUT=0.3 FREEZE_ENCODER_EPOCHS=0 SPLIT_STRATEGY="$split" PIPELINE_MODE=pretrained nohup python chemprop/finetune_chemprop_multitask.py --pretrain_pipeline_mode pretrained --run_id full_run --subset_name new_1M --pretrain_ckpt ~/pepcube_property/pretrained_chemprop/pretrained_model.pt
  MODEL_ARCH=multi_head SPLIT_STRATEGY="$split" PIPELINE_MODE=pretrained nohup python chemprop/evaluate_chemprop_multitask.py --run_id full_run --subset_name new_1M --pretrain_ckpt ~/pepcube_property/pretrained_chemprop/pretrained_model.pt
done
'> ./logs/$(date +%Y-%m-%d)_mh_full_run_final_pretrained_ft_ev_both_splits.log 2>&1 &
```

### Finetuning Multitask Pretrained - Frozen Encoder Run using HP from a small HPO

```bash
ALL_SPLITS="random group"
nohup bash -c '
for split in '"$ALL_SPLITS"'; do
MPNN_DEPTH=6 MPNN_D_H=600 ACCELERATOR=gpu MODEL_ARCH=multi_head FINETUNE_EPOCHS=30 FINETUNE_BATCH=8 FINETUNE_LR=0.0005 FREEZE_ENCODER=true DROPOUT=0.0 SPLIT_STRATEGY="$split" PIPELINE_MODE=pretrained nohup python chemprop/finetune_chemprop_multitask.py --run_id full_run --subset_name new_1M_frozen_hpo_small_batch --pretrain_ckpt ~/pepcube_property/pretrained_chemprop/pretrained_model.pt
MODEL_ARCH=multi_head SPLIT_STRATEGY="$split" PIPELINE_MODE=pretrained nohup python chemprop/evaluate_chemprop_multitask.py --run_id full_run --subset_name new_1M_frozen_hpo_small_batch --pretrain_ckpt ~/pepcube_property/pretrained_chemprop/pretrained_model.pt
done
'> ./logs/$(date +%Y-%m-%d)_mh_full_run_frozen_pretrained_ft_ev_both_splits_hpo_small_batch.log 2>&1 &
```

### Fine-tuning & evaluation, multitask (Chemprop & Chemeleon) - Default

```bash
ALL_SPLITS="random group"
nohup bash -c '
for split in '"$ALL_SPLITS"'; do
  ACCELERATOR=gpu SPLIT_STRATEGY="$split" PIPELINE_MODE=chemprop nohup python chemprop/finetune_chemprop_multitask.py  --run_id full_run
  SPLIT_STRATEGY="$split" PIPELINE_MODE=chemprop nohup python chemprop/evaluate_chemprop_multitask.py  --run_id full_run
done
'> ./logs/$(date +%Y-%m-%d)_mh_full_run_chemprop_default_ft_ev_both_splits.log 2>&1 &
```

### Fine-tuning & evaluation, singletask 

```bash
ALL_MODES="chemprop chemeleon"
ALL_DATASETS="synthesizability chemberta_hemolysis_chemprop_550 even_chemberta_noncanonical_camsol_chemprop_10k"
ALL_SPLITS="random group"

nohup bash -c '
for mode in '"$ALL_MODES"'; do
  for dataset in '"$ALL_DATASETS"'; do
    for split in '"$ALL_SPLITS"'; do
      echo "===Mode=$mode dataset=$dataset split=$split==="
      SPLIT_STRATEGY="$split" PIPELINE_MODE="$mode" ACCELERATOR=gpu nohup python chemprop/finetune_chemprop_singetask.py --dataset_name "$dataset"
      SPLIT_STRATEGY="$split" PIPELINE_MODE="$mode" ACCELERATOR=gpu nohup python chemprop/evaluate_chemprop_singletask.py --dataset_name "$dataset"
    done
  done
done
' > ./logs/$(date +%Y-%m-%d)_finetune_evaluate_singletask_all.log 2>&1 &
echo "PID: $!"
```

### Hyperparameter optimization

Finetuning configurations, from a pretrained checkpoint:

```bash
PIPELINE_MODE=chemprop ACCELERATOR=gpu nohup python chemprop/hpo_finetune_chemprop.py 
--run_id full_run --pretrain_ckpt /home/rbirolo/pepcube_property/pretrained_chemprop/pretrained_model.pt 
> ./logs/$(date +%Y-%m-%d)_full_pretrained.log 2>&1 &
```

Pretraining configurations:

```bash
PIPELINE_MODE=chemprop ACCELERATOR=gpu nohup 
python chemprop/hpo_pretrain_chemprop.py 
--subset_name all_generated_scored_hpo100k_chemberta > ./logs/$(date +%Y-%m-%d)_hpo100K_pretrain_parameters.log 2>&1 &
```

### Prediction

From finetune model that was from a pretrained encoder: (ChemProp, Multitask, Pretrained)

```bash
nohup bash -c '
DATASETS=("./data/experimental_sequences.csv"
"./data/mutated_experimental_peptides_90k.csv"
"./data/converted_literature_sequences.csv")

for input_csv in "${DATASETS[@]}"; do
tag=$(basename "$input_csv" .csv)
SPLIT_STRATEGY=random TORCH_FORCE_WEIGHTS_ONLY_LOAD=0 BASE_DIR=/home/rbirolo/pepcube_property ACCELERATOR=gpu MODEL_ARCH=multi_head nohup python chemprop/predict_chemprop_multitask.py --model_dir /home/rbirolo/pepcube_property/runs/multi_head_pretrained_random/finetune/new_1M/run3_new_chemberta_synth_camsol10k_hemo550/final/ --input_csv "$input_csv" --output_csv "predictions/${tag}_chemprop_pretrained_final.csv"
done
'> ./logs/$(date +%Y-%m-%d)_chemprop_pretrained_predictions.log 2>&1 &
```
From a finetune model that was from a default encoder (ChemProp or Chemeleon):
- For chemeleon, make sure to set PIPELINE_MODE=chemeleon so the proper model is loaded

```bash
nohup bash -c '
DATASETS=("./data/experimental_sequences.csv"
"./data/mutated_experimental_peptides_90k.csv"
"./data/converted_literature_sequences.csv"
)
for input_csv in "${DATASETS[@]}"; do
tag=$(basename "$input_csv" .csv)
SPLIT_STRATEGY=random TORCH_FORCE_WEIGHTS_ONLY_LOAD=0 BASE_DIR=/home/rbirolo/pepcube_property ACCELERATOR=gpu MODEL_ARCH=multi_head nohup python chemprop/predict_chemprop_multitask.py --model_dir runs/multi_head_chemprop_random/finetune/chemprop/run3_new_chemberta_synth_camsol10k_hemo550/final --input_csv "$input_csv" --output_csv "predictions/${tag}_chemprop_default_final.csv"
done
'> ./logs/$(date +%Y-%m-%d)_chemprop_def_predictions_final.log 2>&1 &
```

## Baseline ML
Run from `pepcube_chemprop` env:

```bash
ALL_MODELS="random_forest xgboost catboost svm"
ALL_DATASETS="synthesizability chemberta_hemolysis_chemprop_550 even_chemberta_noncanonical_camsol_chemprop_10k"
ALL_SPLITS="group random"

nohup bash -c '
for model in '"$ALL_MODELS"'; do
  for dataset in '"$ALL_DATASETS"'; do
    for split in '"$ALL_SPLITS"'; do
      nohup python baseline_ml.py --dataset_name "$dataset" --model "$model" --split_strategy "$split"
    done
  done
done
' > ./logs/$(date +%Y-%m-%d)_baseline_ml_single_task_bothsplits.log 2>&1 &
```
## Peptide Generation using p2smi
Note this was run locally, from p2smi's genPep script

#Pretraining Dataset
```bash
python genPeps.py \
  --num 1000000 \
  --min_length 10 \
  --max_length 35 \
  --noncanonical 0.0 \
  --dextro 0.0 \
  --cyclization_constraints None \
  --outfile linear_1M_canonical_peptides.fasta
```

#Non-natural Peptides for CamSol-PTM

```bash
python genPeps.py \
  --num 10000 \                            
  --min_length 10 \
  --max_length 35 \
  --noncanonical 0.2 \
  --dextro 0.0 \
  --cyclization_constraints all \
  --outfile nonnatural_10K_peptides.fasta
```