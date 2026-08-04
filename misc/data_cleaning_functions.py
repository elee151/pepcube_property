import time
from rdkit.Chem import Descriptors
import requests
from chemprop_datacleaning.scripts.smiles_gen_new import *

#using DBAASP's API to fetch modifications for peptides in the hemolysis dataset
BASE_URL = "https://dbaasp.org"
def fetch_peptide(peptide_id, retries = 3):
    url = f"{BASE_URL}/peptides/{peptide_id}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"  [warn] ID {peptide_id} attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)
    return None

def parse_modifications(card):
    mods = []
    for uaa in card.get("unusualAminoAcids", []):
        mods.append({
            "position":            uaa.get("position"),
            "modification_type":   uaa.get("modificationType", {}).get("name"),
            "modification_desc":   uaa.get("modificationType", {}).get("description"),
            "before_modification": uaa.get("beforeModification"),
            "note":                uaa.get("note"),
        })
    return mods

def replace_X_in_sequence(sequence, modifications):
    seq = list(sequence)

    for mod in sorted(modifications, key=lambda m: m["position"] or 0):
        pos = mod["position"]
        if pos is None:
            continue
        idx = pos - 1
        if 0 <= idx < len(seq) and seq[idx] in ("X", "x"):
            uaa_name = mod["modification_type"] or mod["note"] or "UAA"
            seq[idx] = f"[{uaa_name}]"

    return "".join(seq)

def extract_smiles(card):
    smiles_list = card.get("smiles", [])
    if not smiles_list:
        return None

    for s in smiles_list:
        if s.get("manuallyEdited"):
            return s.get("smiles")

    return smiles_list[0].get("smiles")

def enrich_dataframe(df, id_col = "peptide_id", seq_col = "sequence", delay = 0.3):
    seq_expanded_col = []
    modifications_col = []
    smiles_col = []

    for _, row in df.iterrows():
        pid = str(row[id_col])
        seq = str(row[seq_col])

        card = fetch_peptide(pid)

        if card is None:
            print(f"  [error] Could not fetch peptide {pid}, skipping.")
            seq_expanded_col.append(seq)
            modifications_col.append([])
            smiles_col.append(None)
            time.sleep(delay)
            continue

        mods    = parse_modifications(card)
        expanded = replace_X_in_sequence(seq, mods)
        smiles  = extract_smiles(card)

        seq_expanded_col.append(expanded)
        modifications_col.append(mods)
        smiles_col.append(smiles)

        time.sleep(delay)

    result = df.copy()
    result["sequence_expanded"] = seq_expanded_col
    result["modifications"]     = modifications_col
    result["smiles"]            = smiles_col
    return result

def to_canonical_smiles(smiles):
    if not smiles:
        return None
    else:
        mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)

def get_molecular_weight(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return round(Descriptors.ExactMolWt(mol), 4)