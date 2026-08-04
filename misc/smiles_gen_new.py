import pandas as pd
from rdkit import Chem
import re

AMINO_ACID_SMILES = {
    # Standard L-amino acids
    'A': 'N[C@@H](C)C(=O)O',
    'C': 'N[C@@H](CS)C(=O)O',
    'D': 'N[C@@H](CC(=O)O)C(=O)O',
    'E': 'N[C@@H](CCC(=O)O)C(=O)O',
    'F': 'N[C@@H](Cc1ccccc1)C(=O)O',
    'G': 'NCC(=O)O',
    'H': 'N[C@@H](Cc1c[nH]cn1)C(=O)O',
    'I': 'N[C@@H]([C@@H](C)CC)C(=O)O',
    'K': 'N[C@@H](CCCCN)C(=O)O',
    'L': 'N[C@@H](CC(C)C)C(=O)O',
    'M': 'N[C@@H](CCSC)C(=O)O',
    'N': 'N[C@@H](CC(=O)N)C(=O)O',
    'P': 'N1[C@@H](CCC1)C(=O)O',
    'Q': 'N[C@@H](CCC(=O)N)C(=O)O',
    'R': 'N[C@@H](CCCNC(=N)N)C(=O)O',
    'S': 'N[C@@H](CO)C(=O)O',
    'T': 'N[C@@H]([C@@H](C)O)C(=O)O',
    'V': 'N[C@@H](C(C)C)C(=O)O',
    'W': 'N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O',
    'Y': 'N[C@@H](Cc1ccc(O)cc1)C(=O)O',

    # D-amino acids
    'a': 'N[C@H](C)C(=O)O',
    'c': 'N[C@H](CS)C(=O)O',
    'd': 'N[C@H](CC(=O)O)C(=O)O',
    'e': 'N[C@H](CCC(=O)O)C(=O)O',
    'f': 'N[C@H](Cc1ccccc1)C(=O)O',
    'g': 'NCC(=O)O',
    'h': 'N[C@H](Cc1c[nH]cn1)C(=O)O',
    'i': 'N[C@H]([C@H](C)CC)C(=O)O',
    'k': 'N[C@H](CCCCN)C(=O)O',
    'l': 'N[C@H](CC(C)C)C(=O)O',
    'm': 'N[C@H](CCSC)C(=O)O',
    'n': 'N[C@H](CC(=O)N)C(=O)O',
    'p': 'N1[C@H](CCC1)C(=O)O',
    'q': 'N[C@H](CCC(=O)N)C(=O)O',
    'r': 'N[C@H](CCCNC(=N)N)C(=O)O',
    's': 'N[C@H](CO)C(=O)O',
    't': 'N[C@H]([C@H](C)O)C(=O)O',
    'v': 'N[C@H](C(C)C)C(=O)O',
    'w': 'N[C@H](Cc1c[nH]c2ccccc12)C(=O)O',
    'y': 'N[C@H](Cc1ccc(O)cc1)C(=O)O',

    # AZA-amino acids
    'azaA': 'NNC(C)C(=O)O',
    'azaC': 'NNC(CS)C(=O)O',
    'azaD': 'NNC(CC(=O)O)C(=O)O',
    'azaE': 'NNC(CCC(=O)O)C(=O)O',
    'azaF': 'NNC(Cc1ccccc1)C(=O)O',
    'azaG': 'NNCC(=O)O',
    'azaH': 'NNC(Cc1c[nH]cn1)C(=O)O',
    'azaI': 'NNC(C(C)CC)C(=O)O',
    'azaK': 'NNC(CCCCN)C(=O)O',
    'azaL': 'NNC(CC(C)C)C(=O)O',
    'azaM': 'NNC(CCSC)C(=O)O',
    'azaN': 'NNC(CC(=O)N)C(=O)O',
    'azaP': 'NN1CCCC1C(=O)O',
    'azaQ': 'NNC(CCC(=O)N)C(=O)O',
    'azaR': 'NNC(CCCNC(=N)N)C(=O)O',
    'azaS': 'NNC(CO)C(=O)O',
    'azaT': 'NNC(C(C)O)C(=O)O',
    'azaV': 'NNC(C(C)C)C(=O)O',
    'azaW': 'NNC(Cc1c[nH]c2ccccc12)C(=O)O',
    'azaY': 'NNC(Cc1ccc(O)cc1)C(=O)O',

    #replacement DBAASP UAA library of modifications
    'Me-Phe': 'CN[C@@H](Cc1ccccc1)C(O)=O',
    '3,4-OH-ARG': 'N[C@@H](C(O)C(O)CNC(N)=N)C(=O)O',
    '5-OH-LYS': 'NCC(O)CC[C@H](N)C(=O)O',
    'HPro': 'OC(=O)C1CCCCN1',
    'NLys': 'C(CCNCC(=O)O)CN',
    'NNLe': 'CCCCNCC(O)=O',
    'Nap': 'NCCCNCC(O)=O',
    'HLeu': 'CC(C)CC[C@H](N)C(O)=O',
    'AGL': 'OC(=O)CNCC=C',
    '3-NH2-14-Me-C16': 'CCC(C)CCCCCCCCCCC(N)CC(O)=O',
    '6F-LEU': 'N[C@@H](CC(C(F)(F)F)C(F)(F)F)C(O)=O',
    'Tic': 'OC(=O)C1NCCc2ccccc12',
    'Et-Cys': 'CCN[C@@H](CS)C(=O)O',
    'D-NLE': 'CC(C)C[C@H](C(=O)O)NC',
    'AdaGly': 'OC(=O)CNC12CC3CC(CC(C3)C1)C2',
    '4-NO2-PHE': 'N[C@@H](Cc1ccc(cc1)[N+]([O-])=O)C(O)=O',
    'Œ≤Nspe': 'N([C@@H](C)c1ccccc1)CCC(=O)',
    'IAA-Cys': 'N[C@@H](CS)C(=O)O.NC(=O)CI',
    'D-HSer': 'N[C@H](CCO)C(=O)O',
    'D-2-NAL': 'C1=CC=C2C=C(C=CC2=C1)C[C@@H](C(=O)O)N',
    'Ahx': 'NCCCCCC(O)=O',
    '5-OH-TRP': 'N[C@@H](Cc1c[nH]c2ccc(O)cc12)C(O)=O',
    'Ac6c': 'NC1(CCCCC1)C(O)=O',
    'DIP': 'N[C@@H](C(c1ccccc1)c2ccccc2)C(=O)O',
    'MET(O)': 'C[S](=O)CC[C@H](N)C(O)=O',
    'D-ORN': 'CN[C@@H](CC1=CC=C(C=C1)O)C(=O)O',
    'THR-Cl': 'N[C@@H]([C@H](O)CCl)C(O)=O',
    'Allo-Thr': 'C[C@H](O)[C@H](N)C(=O)O',
    'D-TERT BU TRP': 'CC(C)(C)C1=CC(C(C)(C)C)=C2NC(C(C)(C)C)=C(C[C@H](N)C(=O)O)C2=C1',
    'NVal': 'N[C@@H](CCC)C(=O)O',
    'Sar': 'CNCC(O)=O',
    'Nnm': 'OC(=O)CNCc1cccc2ccccc12',
    '8-Aoc': 'NCCCCCCCC(O)=O',
    '3-OH-Asp': 'N[C@@H](C(O)C(O)=O)C(=O)O',
    'DHB': 'CC=C(C(=O)O)N',
    'Œ≤ALA': 'NCCC(=O)O',
    'βALA': 'NCCC(=O)O',
    '2-NAL': 'C[C@H](Nc1ccc2ccccc2c1)C(=O)O',
    '(S,S)-ACPC': 'N[C@H]1CCC[C@@H]1C(=O)O',
    'HTrp': 'CCCCCCCCNCC(=O)O',
    'Me-Ser': 'CC(CO)(C(=O)O)N',
    'Npm': 'C1=CC=C(C=C1)CNCC(=O)[O-]',
    'Phg': 'C(=O)OCNc1ccccc1',
    'AIB': 'CC(C)(C(=O)O)N',
    'Ada-Ala': 'C[C@H](NC12CC3CC(CC(C3)C1)C2)C(O)=O',
    'DOPA': 'N[C@@H](Cc1ccc(O)c(O)c1)C(=O)O',
    'R-ALA-7-oct': 'C[C@@](CCCCCCC=C)(C(=O)O)N',
    'hArg': 'N[C@@H](CCCCNC(N)=N)C(O)=O',
    'Me-LYS': 'CN[C@@H](CCCCN)C(=O)O',
    'BIP': 'C1=CC=C(C=C1)C2=CC=C(C=C2)C[C@@H](C(=O)O)N',
    'ORN': 'N[C@@H](CCCN)C(=O)O',
    'Nspe': 'C[C@H](NCC(=O)O)c1ccccc1',
    'GABA': 'NCCCC(=O)O',
    'Oic': 'OC(=O)C1CC2CCCCC2N1',
    '6-Br-trp': 'N[C@@H](Cc1c[nH]c2cc(Br)ccc12)C(=O)O',
    'Eps-LYS': 'CCCCCC(C(=O)NC)N',
    'S-ALA-4-pen': 'C[C@](CCCC=C)(C(=O)O)N',
    '2-Aoc': 'CCCCCCC(N)C(O)=O',
    'DAB': 'CCC(N)(N)C([O-])=O',
    '12-NH2-C12': 'NCCCCCCCCCCCC(O)=O',
    'ABU': 'C(CCCN)CCC(=O)O',
    'Npm(p-CH3)': 'Cc1ccc(CNCC(O)=O)cc1',
    '1-NAL': 'C[C@H](Nc1cccc2ccccc12)C(O)=O',
    'Cha': 'N[C@@H](C1CCCCC1)C(=O)O',
    'Chg': 'N[C@@H](C1CCCCC1)C(=O)O',
    'Acm-Cys': 'CC(=O)NCSC[C@H](N)C(O)=O',
    'HSer': 'N[C@@H](CCO)C(=O)O',
    '4,5-OH-LYS': 'NCC(O)C(O)C[C@H](N)C(=O)O',
    'NLE': 'N[C@@H](CCCC)C(=O)O',
    '4-3Fm-PHE': 'N[C@@H](Cc1ccc(cc1)C(F)(F)F)C(O)=O',
    '3-Abz': 'C1=CC(=CC(=C1)N)C(=O)O',
    'D-1-NAL': 'C1[C@@H](O1)CNC(=O)C2=CC=C(S2)Cl',
    '7-NH2-C7': 'NCCCCCCC(=O)O',
    'Ndpe': 'OC(=O)CNCC(c1ccccc1)c2ccccc2',
    'TERT BU PHE': 'CC(C)(C)c1ccc(C[C@H](N)C(O)=O)cc1',
    'ivDde':'N[C@@H](CNC(=C1C(=O)CC(C)(C)CC1=O)CC(C)C)C(=O)O',
    'MSE': 'N[C@@H](CC[Se]C)C(=O)O',
    'MLY': 'N[C@@H](CCCCN(C)C)C(=O)O',
    'PTR': 'N[C@@H](Cc1ccc(OP(=O)(O)O)cc1)C(=O)O',
    'SEP': 'N[C@@H](COP(=O)(O)O)C(=O)O',
    'TPO': 'N[C@@H]([C@H](C)OP(=O)(O)O)C(=O)O',
    'MLZ': 'N[C@@H](CCCCNC)C(=O)O',
    'ALY': 'N[C@@H](CCCCNC(=O)C)C(=O)O',
    'HIC': 'N[C@@H](Cc1c[nH]cn1C)C(=O)O',
    'HYP': 'N1[C@@H](C[C@H](O)C1)C(=O)O',
    'M3L': 'N[C@@H](CCCC[N+](C)(C)C)C(=O)O',
    'PFF': 'N[C@@H](Cc1ccc(F)cc1)C(=O)O',
    'MHO': 'N[C@@H](CCS(=O)C)C(=O)O',

}

N_TERMINAL_MODS = {
    'ACT':     'CC(=O)',
    'CH3CO':  'CC(=O)',
    'formyl': 'C(=O)',
    'Boc':    'CC(C)(C)OC(=O)',
    'Fmoc':   'O=C(OCC1c2ccccc2-c2ccccc21)',
    'Cbz':    'O=C(OCc1ccccc1)',
    '3,4-OH-C16' : 'CCCCCCCCCCCCCC(O)C(O)CC(=O)O',
    'C10' : 'CCCCCCCCCC(=O)O',
    'C12' : 'CCCCCCCCCCCC(=O)O',
    'C14' : 'CCCCCCCCCCCCCC(=O)O',
    'C16' : 'CCCCCCCCCCCCCCCC(=O)O',
    'C18' : 'CCCCCCCCCCCCCCCCCC(=O)O',
    'C5' : 'CCCCC(=O)O',
    'C6' : 'CCCCCC(=O)O',
    'C8' : 'CCCCCCCC(=O)O',
    'CAA' : 'C(C(=O)O)Cl',
    'DNS' : 'CN(C)C1=CC=CC2=C1C=CC=C2S(=O)(=O)Cl',
    'FOR' : 'C(=O)O',
    'OIle' : 'CC[C@@H](C)[C@H](C(=O)O)O',
}

C_TERMINAL_MODS = {
    'AMD':        'C(N)=O',
    'NH2':        'C(N)=O',
    'NHCH2CH2CH3':'C(NCCC)=O',
    'NHCH2CH3':   'C(NCC)=O',
    'OMe':        'C(=O)OC',
    'OEt':        'C(=O)OCC',
    'OtBu':       'C(=O)OC(C)(C)C',
    'PHEol':      'C[C@@H](CO)N',
    'ETA':        'C(CO)N',
    'VALol':      'CC(C)C(CO)N',
    'EN':         'C(CN)N',
    'AM':         'C(=O)NCC',
    'C16H33-NH2': 'CCCCCCCCCCCCCCCCN',
}

BOND_TYPES = {
    'CH2NH':    'reduced_amide',
    'CH2S':     'thiomethylene',
    'CH2OCONH': 'reduced_carbamate',
}

_TOKEN_RE = re.compile(r'\[([^\]]+)\]|([A-Za-z])')

def tokenize_sequence(sequence: str) -> list[tuple[str, str]]:
    """Return a list of (role, token) pairs for a peptide sequence string."""
    parts = [m.group(1) if m.group(1) is not None else m.group(0) for m in _TOKEN_RE.finditer(sequence)]

    components = []
    for i, part in enumerate(parts):
        clean = part.strip('[]')
        if clean in BOND_TYPES:
            components.append(('bond', clean))
        elif i == 0 and clean in N_TERMINAL_MODS:
            components.append(('n_mod', clean))
        elif i == len(parts) - 1 and clean in C_TERMINAL_MODS:
            components.append(('c_mod', clean))
        elif clean in AMINO_ACID_SMILES:
            components.append(('aa', clean))
        else:
            components.append(('unknown', clean))

    return components

def peptide_smiles(sequence):
    if pd.isna(sequence):
        return None

    try:
        components = tokenize_sequence(sequence)
        if not components:
            return None

        amino_acids, bond_mods = [], {}
        n_term_mod = c_term_mod = None
        aa_count = 0

        for comp_type, value in components:
            if comp_type == 'aa':
                amino_acids.append(value)
                aa_count += 1
            elif comp_type == 'bond':
                bond_mods[aa_count - 1] = value
            elif comp_type == 'n_mod':
                n_term_mod = value
            elif comp_type == 'c_mod':
                c_term_mod = value
            else:  # unknown
                print(f"Unknown component '{value}' in sequence: {sequence}")
                return None

        if not amino_acids:
            return None

        smi = AMINO_ACID_SMILES[amino_acids[0]]

        if n_term_mod:
            smi = N_TERMINAL_MODS[n_term_mod] + smi

        for i in range(1, len(amino_acids)):
            next_smi = AMINO_ACID_SMILES[amino_acids[i]]
            bond_type = bond_mods.get(i - 1, 'normal')
            last_idx = smi.rfind('C(=O)O')

            if last_idx == -1:
                print(f"Cannot find C-terminus building position {i} of: {sequence}")
                return None

            if bond_type == 'normal':
                smi = smi[:last_idx] + 'C(=O)' + smi[last_idx + 6:] + next_smi
            elif bond_type == 'CH2NH':
                smi = smi[:last_idx] + 'C' + smi[last_idx + 6:] + next_smi
            elif bond_type == 'CH2S':
                smi = smi[:last_idx] + 'CS' + smi[last_idx + 6:] + next_smi
            elif bond_type == 'CH2OCONH':
                smi = smi[:last_idx] + 'COC(=O)' + smi[last_idx + 6:] + next_smi

        if c_term_mod:
            last_idx = smi.rfind('C(=O)O')
            if last_idx != -1:
                replacement = C_TERMINAL_MODS[c_term_mod]

                # replace entire terminal motif, not append
                smi = smi[:last_idx] + replacement + smi[last_idx + len('C(=O)O'):]

        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol) if mol else None

    except Exception as e:
        print(f"Error processing '{sequence}': {e}")
        return None