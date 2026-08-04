from Bio.SeqUtils.ProtParam import ProteinAnalysis
from chemprop_datacleaning.scripts.data_cleaning_functions import *

# Synthesizability Dif - aa coefficients determined by excel sheet
AA_COEFFICIENTS = {
    "A": 1.34,
    "R": 0.46,
    "N": 0.97,
    "D": 0.63,
    "C": 1.09,
    "Q": 0.79,
    "E": 1.10,
    "G": 0.81,
    "H": 0.64,
    "I": 1.58,
    "L": 1.20,
    "K": 1.31,
    "M": 1.15,
    "F": 1.07,
    "P": 0.26,
    "S": 0.69,
    "T": 1.15,
    "W": 1.01,
    "Y": 1.12,
    "V": 1.77,
    "X": 0.40,
    "J": 0.40,
    "Z": 0.40,
}

DEFAULT_COEFFICIENT = 0.90
TAIL_PLACEHOLDER    = 0.88

def get_aa_coefficient(aa, warn = True):
    aa = aa.upper()
    if aa in AA_COEFFICIENTS:
        return AA_COEFFICIENTS[aa]
    if warn and aa.isalpha():
        print(f"'{aa}' is not defined: using average value ({DEFAULT_COEFFICIENT})")
    return DEFAULT_COEFFICIENT

def score_sequence(sequence):
    """
    Difficulty Algorithm from Excel Macro:
    - For each amino acid at position n in the sequence, pull (5-residue window) starting at that position.
    - Sum the difficulty coefficients for each residue in the window.
    - Divide by 5 to get the average to get difficulty coupling score for position n.
    - For the last 4 residues (where a full pentamer cannot be formed), assign score of 0.88.
    """

    sequence = sequence.strip().upper()
    seq_len  = len(sequence)
    scores   = []

    for n in range(seq_len):
        is_tail = n >= seq_len - 4
        if is_tail:
            scores.append(TAIL_PLACEHOLDER)
        else:
            pentamer = sequence[n : n + 5]
            difcoef  = sum(get_aa_coefficient(aa) for aa in pentamer)
            scores.append(round(difcoef / 5, 2))
    return scores

def average_difficulty(sequence):
    scores = score_sequence(sequence)
    avg = round(sum(scores) / len(scores), 4) if scores else float("nan")
    return avg

def calculate_difficulty_df(df, seq_column):
    """produces a df with scores of synthesis difficulty per aa, with average score in another column"""
    sequences = df[seq_column]
    if isinstance(sequences, str):
        sequences = [sequences]

    rows = []
    for seq in sequences:
        seq      = seq.strip().upper()
        scores   = score_sequence(seq)
        avg      = average_difficulty(seq)
        rows.append({"sequence": seq, "aa_scores": scores, "average": avg})

    return pd.DataFrame(rows, columns=["sequence", "average", "aa_scores"])

def calculate_peptide_metrics(df, class_col = 'class' ,seq_col='Peptide Sequence', metrics='all', target_ph=7.4):
    """Using biopython to calculate various peptide sequences scores"""
    mask = df[class_col] == 'generated_canonical'
    available = {'gravy', 'instability', 'charge'}
    selected = list(available) if metrics == 'all' else [m for m in metrics if m in available]

    if not selected or df.empty:
        return df

    def get_scores(seq):
        if pd.isna(seq) or str(seq).strip() == "":
            return [None] * len(selected)

        try:
            pa = ProteinAnalysis(str(seq))
            res_dict = {
                'gravy': pa.gravy(),
                'instability': pa.instability_index(),
                'charge': pa.charge_at_pH(target_ph)
            }
            return [res_dict[m] for m in selected]

        except:
            return [None] * len(selected)

    col_names = [f'charge_at_pH_{target_ph}' if m == 'charge' else m for m in selected]
    df[col_names] = None

    df.loc[mask, col_names] = df.loc[mask, seq_col].apply(get_scores).apply(pd.Series).values

    return df


def sample_generated_by_cluster_type(df, total_n, in_space_prop, cluster_col, class_col='class'):
    """sampling generated peptides based on various clustering methods"""
    exp_classes = ['hemolysis_smiles', 'stability_smiles', 'synthesize_smiles']

    exp_clusters = df[df[class_col].isin(exp_classes)][cluster_col].unique()
    df_gen = df[df[class_col] == 'generated_canonical']
    gen_in_exp_space = df_gen[df_gen[cluster_col].isin(exp_clusters)]
    gen_out_of_space = df_gen[~df_gen[cluster_col].isin(exp_clusters)]

    n_in_space = int(total_n * in_space_prop)
    n_out_space = total_n - n_in_space

    # in experimental clusters sampling using even sampling across exp clusters ex. 10 from 50 clusters
    in_space_samples = []
    if not gen_in_exp_space.empty:
        n_per_cluster = n_in_space // len(exp_clusters)
        remainder = n_in_space % len(exp_clusters)

        for i, cluster in enumerate(exp_clusters):
            c_data = gen_in_exp_space[gen_in_exp_space[cluster_col] == cluster]
            target = n_per_cluster + (1 if i < remainder else 0)
            if not c_data.empty:
                in_space_samples.append(c_data.sample(min(len(c_data), target), random_state=42))


    # if the proportion is higher than the number sampled using even sampling,
    # additional ones are added randomly from exp.
    already_sampled_idx = (
        pd.concat(in_space_samples).index if in_space_samples else pd.Index([])
    )
    n_collected = len(already_sampled_idx)
    deficit = n_in_space - n_collected

    if deficit > 0:
        remaining_pool = gen_in_exp_space.drop(index=already_sampled_idx, errors='ignore')
        if not remaining_pool.empty:
            in_space_samples.append(
                remaining_pool.sample(min(len(remaining_pool), deficit), random_state=42)
            )

    # generated peptides (out of experimental peptide space) peptides only
    out_space_samples = []
    if not gen_out_of_space.empty and n_out_space > 0:
        out_space_samples.append(
            gen_out_of_space.sample(min(len(gen_out_of_space), n_out_space), random_state=42)
        )

    final_df = pd.concat(in_space_samples + out_space_samples).reset_index(drop=True)
    return final_df
