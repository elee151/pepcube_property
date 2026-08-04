import random
import itertools
import pandas as pd
from pepcube_property.misc.smiles_gen_new import *

"""
Example usage

ncaa = ['NLE','Cha','AIB','ivDde','Chg','GABA']
df = mutate(starter_seqs["plain_seq"],
            n_mutations=(3, 5), 
            ncaa=ncaa, 
            cap=100000, 
            aa_pool='different_class', 
            max_ncaa=2, 
            position_window=(0,5))
"""
# tokenize_sequence pulled from smiles_gen_new
AA_CLASSES = {
    'aromatic':    ['F', 'W', 'Y'],
    'basic':       ['K', 'R', 'H'],
    'acidic':      ['D', 'E'],
    'polar':       ['S', 'T', 'N', 'Q','C'],
    'hydrophobic': ['A', 'I', 'L', 'V', 'M'],
    'small':       ['G', 'P'],
}
AA_TO_CLASS = {aa: cls for cls, aas in AA_CLASSES.items() for aa in aas}
ALL_CANONICAL = [aa for aas in AA_CLASSES.values() for aa in aas]

def build_pool(current_aa, aa_pool):
    current_class = AA_TO_CLASS.get(current_aa)
    if aa_pool == 'same_class' and current_class:
        return [a for a in AA_CLASSES[current_class] if a != current_aa]
    if aa_pool == 'different_class' and current_class:
        return [a for a in ALL_CANONICAL if AA_TO_CLASS.get(a) != current_class]
    return [a for a in ALL_CANONICAL if a != current_aa]


def sample_positions(aa_indices, n_mut, n_combos):
    """1/3 clustered (random window start + size), 1/3 evenly spread
    (random step + offset), 1/3 fully random."""
    n = len(aa_indices)
    if n < n_mut:
        return []

    results = set()
    per_strategy = max(1, n_combos // 3)

    # clustered: set window of positions, start and size randomized each time
    tries = 0
    target = per_strategy
    while len(results) < target and tries < per_strategy * 10:
        tries += 1
        window_size = random.randint(n_mut, n)
        if n <= window_size:
            window = aa_indices
        else:
            start = random.randint(0, n - window_size)
            window = aa_indices[start:start + window_size]
        results.add(tuple(sorted(random.sample(window, n_mut))))
    n_after_clustered = len(results)

    # spread: evenly spaced, step/offset/jitter randomized each draw
    tries = 0
    target = n_after_clustered + per_strategy
    while len(results) < target and tries < per_strategy * 10:
        tries += 1
        step = n / n_mut
        jitter_span = max(1, int(step // 2))
        offset = random.uniform(0, step)
        chosen = []
        for i in range(n_mut):
            base_idx = min(int(offset + i * step), n - 1)
            idx = max(0, min(n - 1, base_idx + random.randint(-jitter_span, jitter_span)))
            chosen.append(aa_indices[idx])
        combo = tuple(sorted(set(chosen)))
        if len(combo) == n_mut:
            results.add(combo)
    n_after_spread = len(results)

    # random: completely random across the entire peptide
    target = n_after_spread + per_strategy
    tries = 0
    while len(results) < target and tries < per_strategy * 20:
        tries += 1
        results.add(tuple(sorted(random.sample(aa_indices, n_mut))))

    return list(results)


def mutate(
    starting_seqs,
    n_mutations,
    ncaa,
    cap=10000,
    aa_pool='all',
    max_ncaa=None,
    position_window=None,
) -> pd.DataFrame:
    """
    starting_seqs: list of peptide sequence strings
    n_mutations : int or (min, max) tuple, number of mutated positions
    ncaa : list of non-canonical AA keys usable as replacements
    cap : max total sequences generated
    aa_pool : 'all' | 'same_class' | 'different_class'
    max_ncaa : max ncaa substitutions per mutant (None = unlimited)
    position_window  : (0, 5) = full peptide excluding the last 5 AAs. None = full peptide.
    """
    if aa_pool not in ('all', 'same_class', 'different_class'):
        raise ValueError("aa_pool must be 'all', 'same_class', or 'different_class'")

    n_range = (n_mutations, n_mutations) if isinstance(n_mutations, int) else n_mutations

    # build one bucket per (seq, n_mut) combo up front
    buckets = []
    for seq in starting_seqs:
        tokens = tokenize_sequence(seq)
        all_aa_indices = [i for i, (role, _) in enumerate(tokens) if role == 'aa']
        if not all_aa_indices:
            continue

        if position_window is not None:
            n_offset, c_offset = position_window
            last_pos = len(all_aa_indices) - 1
            allowed = range(n_offset, last_pos - c_offset + 1)
            aa_indices = [idx for pos, idx in enumerate(all_aa_indices) if pos in allowed]
        else:
            aa_indices = all_aa_indices

        if not aa_indices:
            continue

        for n_mut in range(n_range[0], n_range[1] + 1):
            if n_mut <= len(aa_indices):
                buckets.append({
                    'seq': seq, 'tokens': tokens, 'all_aa_indices': all_aa_indices,
                    'aa_indices': aa_indices, 'n_mut': n_mut,
                })

    if not buckets:
        return pd.DataFrame([])

    rows = []
    seen_global = set()
    exhausted = set()

    # iterates over catogories until the cap limit is hit
    while len(rows) < cap and len(exhausted) < len(buckets):
        active = len(buckets) - len(exhausted)
        batch_size = max(1, (cap - len(rows)) // active)

        for bi, b in enumerate(buckets):
            if bi in exhausted or len(rows) >= cap:
                continue

            seq, tokens = b['seq'], b['tokens']
            all_aa_indices, aa_indices, n_mut = b['all_aa_indices'], b['aa_indices'], b['n_mut']

            pos_combos = sample_positions(aa_indices, n_mut, batch_size)
            if not pos_combos:
                exhausted.add(bi)
                continue

            # cap how many rows any single position combo can contribute
            per_combo_budget = max(1, batch_size // len(pos_combos))
            added = 0

            for pos_combo in pos_combos:
                if added >= batch_size or len(rows) >= cap:
                    break

                pools = [build_pool(tokens[idx][1], aa_pool) + list(ncaa) for idx in pos_combo]
                total_combos = 1
                for p in pools:
                    total_combos *= len(p)

                if total_combos <= 200:
                    replacement_iter = itertools.product(*pools)
                else:
                    replacement_iter = (tuple(random.choice(p) for p in pools) for _ in range(200))

                combo_added = 0
                for replacements in replacement_iter:
                    if combo_added >= per_combo_budget or added >= batch_size or len(rows) >= cap:
                        break

                    if max_ncaa is not None and sum(r in ncaa for r in replacements) > max_ncaa:
                        continue

                    new_tokens = list(tokens)
                    for idx, aa in zip(pos_combo, replacements):
                        new_tokens[idx] = ('aa', aa)

                    mutated_seq = ''.join(v if len(v) == 1 else f'[{v}]' for _, v in new_tokens)

                    if mutated_seq == seq or mutated_seq in seen_global:
                        continue
                    seen_global.add(mutated_seq)

                    rows.append({
                        'starter_seq': seq,
                        'n_mutations': n_mut,
                        'mutated_seq': mutated_seq,
                        'positions': tuple(all_aa_indices.index(i) for i in pos_combo),
                    })
                    added += 1
                    combo_added += 1

            if added == 0:
                exhausted.add(bi)

    # then draws fully random mutations across all buckets until cap is hit if others are exhausted
    if len(rows) < cap:
        max_stall = 5000
        stall = 0
        while len(rows) < cap and stall < max_stall:
            b = random.choice(buckets)
            seq, tokens = b['seq'], b['tokens']
            all_aa_indices, aa_indices, n_mut = b['all_aa_indices'], b['aa_indices'], b['n_mut']

            pos_combo = tuple(sorted(random.sample(aa_indices, n_mut)))
            replacements = tuple(
                random.choice(build_pool(tokens[idx][1], aa_pool) + list(ncaa))
                for idx in pos_combo
            )

            if max_ncaa is not None and sum(r in ncaa for r in replacements) > max_ncaa:
                stall += 1
                continue

            new_tokens = list(tokens)
            for idx, aa in zip(pos_combo, replacements):
                new_tokens[idx] = ('aa', aa)
            mutated_seq = ''.join(v if len(v) == 1 else f'[{v}]' for _, v in new_tokens)

            if mutated_seq == seq or mutated_seq in seen_global:
                stall += 1
                continue

            seen_global.add(mutated_seq)
            rows.append({
                'starter_seq': seq,
                'n_mutations': n_mut,
                'mutated_seq': mutated_seq,
                'positions': tuple(all_aa_indices.index(i) for i in pos_combo),
            })
            stall = 0

    return pd.DataFrame(rows)