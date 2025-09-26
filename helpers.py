import pandas as pd
import numpy as np
import re
from scipy import stats
from itertools import combinations

# --- Parse sheet label into a date ---
def parse_sheet_label_to_date(label: str):
    for dayfirst in (True, False):
        dt = pd.to_datetime(label, errors="coerce", dayfirst=dayfirst)
        if pd.notna(dt):
            return dt
    return None

# --- Put labels in chronological order ---
def chronological_labels(labels):
    pairs = [(lab, parse_sheet_label_to_date(lab)) for lab in labels]
    pairs_sorted = sorted(
        pairs, key=lambda x: (pd.isna(x[1]), x[1] if pd.notna(x[1]) else pd.Timestamp.max)
    )
    return [p[0] for p in pairs_sorted]

# --- Generate compact letter display (CLD) ---
def generate_cld_overlap(means, mse, df_error, alpha, rep_counts, a_is_lowest=True):
    trts = list(means.index)
    letters = {t: set() for t in trts}

    nsd = pd.DataFrame(False, index=trts, columns=trts)
    for t in trts:
        nsd.loc[t, t] = True

    t_crit = stats.t.ppf(1 - alpha/2, df_error) if df_error > 0 else np.nan
    for a, b in combinations(trts, 2):
        n1, n2 = rep_counts.get(a, 1), rep_counts.get(b, 1)
        if n1 > 0 and n2 > 0 and pd.notna(mse) and pd.notna(t_crit):
            lsd_pair = t_crit * np.sqrt(mse * (1/n1 + 1/n2))
            diff = abs(means[a] - means[b])
            if diff <= lsd_pair:
                nsd.loc[a, b] = True
                nsd.loc[b, a] = True

    order = means.sort_values(ascending=a_is_lowest).index
    groups = []
    next_letter_code = ord("a")

    for t in order:
        joined_any = False
        for g in groups:
            if all(nsd.loc[t, m] for m in g["members"]):
                letters[t].add(g["letter"])
                g["members"].append(t)
                joined_any = True
        if not joined_any:
            new_letter = chr(next_letter_code)
            groups.append({"letter": new_letter, "members": [t]})
            letters[t].add(new_letter)
            next_letter_code += 1

        # Expand groups if needed
        changed = True
        while changed:
            changed = False
            for g in groups:
                for cand in trts:
                    if g["letter"] not in letters[cand]:
                        if all(nsd.loc[cand, m] for m in g["members"]):
                            letters[cand].add(g["letter"])
                            g["members"].append(cand)
                            changed = True

    return {t: "".join(sorted(v)) for t, v in letters.items()}, nsd

# --- Generate safe Streamlit widget keys ---
def safe_key(base, assess):
    safe = re.sub(r"\\W+", "_", str(assess))
    return f"{base}_{safe}"
