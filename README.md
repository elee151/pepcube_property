# pepcube_property

Peptide property prediction pipeline: synthesizability, hemolysis, and
CamSol solubility, across Chemprop (MPNN) and transformer (ChemBERTa /
PepDoRA) model families, and ML
baselines.

## Setup

Chemprop and the transformers (ChemBERTa/PepDoRA/LoRA) are kept in
two separate environments

```bash
git clone pepcube_property
cd ..

# Chemprop environment
conda env create -f pepcube_property/environment_chemprop.yml
conda activate pepcube_chemprop

# Transformer environment
conda env create -f pepcube_property/environment_transformer.yml
conda activate pepcube_transformer

# One-time set up inside the transformer env: download the ChemBERTa/PepDoRA
# weights the transformer pipeline expects
conda activate pepcube_transformer
cd pepcube_property
python -m setup_hf_cache
```
Note that Chemeleon model weights need to be downloaded into the base_dir from https://zenodo.org/records/15460715.
## Results
Results are split by model family into three independent CSVs
under `<BASE_DIR>/results/`:

- `chemprop_results.csv`
- `transformer_results.csv`
- `baseline_results.csv`

Each row is either a CV row (`mean_*`/`std_*`) or a test-set evaluation (`rmse`/`mae`).
Columns:

```
run_timestamp, run_date, job_id, model_arch, run_name, head, dataset_name,
pipeline_mode, subset_label, split_strategy,
mean_rmse, std_rmse, mean_mae, std_mae, mean_r2, std_r2,
mean_pearson_r, std_pearson_r, mean_spearman_r, std_spearman_r,
rmse, mae, r2, pearson_r, spearman_r
```
