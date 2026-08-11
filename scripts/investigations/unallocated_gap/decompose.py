"""Decompose the 348M unallocated-gap population by root cause, stratum level."""
import pathlib
import pandas as pd
import numpy as np

REPO = "/Users/kevin/Documents/ProgrammingIsFun/ALLFED/water-access"
md = pd.read_csv(f"{REPO}/data/processed/merged_data.csv")
uni = pd.read_csv(f"{REPO}/data/processed/semi-processed/jmp_source_breakdown_unified.csv")
who = pd.read_csv(f"{REPO}/data/original_data/WHO Household Water Data - 2023 Data.csv", low_memory=False)

who = who.replace({"<1": 0, ">99": 100, "-": pd.NA})
for c in ["URBANPiped", "RURALPiped", "URBANNon-piped", "RURALNon-piped"]:
    who[c] = pd.to_numeric(who[c], errors="coerce")
who["Year"] = pd.to_numeric(who["Year"], errors="coerce")

# alpha3 mapping identical in spirit to the parser
import pycountry
MANUAL = {"Channel Islands": "CHI", "United States Virgin Islands": "VIR",
          "Wallis and Futuna Islands": "WLF", "Democratic Republic of the Congo": "COD",
          "China, Hong Kong SAR": "HKG", "China, Macao SAR": "MAC",
          "Republic of Korea": "KOR", "Curaçao": "CUW",
          "Sint Maarten (Dutch part)": "SXM",
          "Democratic People's Republic of Korea": "PRK", "Niger": "NER"}
def to_a3(name):
    if name in MANUAL: return MANUAL[name]
    try: return pycountry.countries.search_fuzzy(str(name).split("(")[0].strip())[0].alpha_3
    except LookupError: return None
who["alpha3"] = who["Country"].map(to_a3)

# parser's reference: latest year unconditionally
latest = who.sort_values(["Country", "Year"], ascending=[True, False]).drop_duplicates("Country", keep="first")
dupes = latest[latest.alpha3.duplicated(keep=False)][["Country", "alpha3"]]
if len(dupes):
    print("WARN duplicate alpha3 in WHO reference:\n", dupes.to_string(index=False))
latest = latest.drop_duplicates("alpha3", keep="first")
ref = latest.set_index("alpha3")[["Year", "URBANPiped", "RURALPiped", "URBANNon-piped", "RURALNon-piped"]]

cats = ["Packaged", "Delivered", "Borehole", "OtherUnpiped"]
md["urb_frac"] = md["% urban"] / 100

uni_idx = uni.set_index("alpha3")
rows = []
for _, r in md.iterrows():
    a3 = r["alpha3"]
    origin = uni_idx.loc[a3, "data_origin"] if a3 in uni_idx.index else "missing_from_unified"
    for s, frac in (("URBAN", r["urb_frac"]), ("RURAL", 1 - r["urb_frac"])):
        pop = r["Population"] * frac
        P = r[f"{s}Piped"]
        K = sum(r[f"{s}{c}"] for c in cats)
        gap = max(0.0, 100 - P - K)
        # unified NaN pattern for this stratum
        n_nan = 0
        if a3 in uni_idx.index:
            n_nan = sum(pd.isna(uni_idx.loc[a3, f"{s}{c}"]) for c in cats)
        # parser reference state
        ref_piped = ref.loc[a3, f"{s}Piped"] if a3 in ref.index else np.nan
        ref_np = ref.loc[a3, f"{s}Non-piped"] if a3 in ref.index else np.nan
        ref_none = a3 in ref.index and pd.isna(ref_piped)
        no_who = a3 not in ref.index
        # decomposition vs WHO ladder (only meaningful when ref exists)
        ladder_resid = np.nan  # 100 - P - NP
        subcat_short = np.nan  # NP - K
        if not pd.isna(ref_np) and not pd.isna(ref_piped):
            ladder_resid = 100 - ref_piped - ref_np
            subcat_short = ref_np - K
        rows.append(dict(alpha3=a3, stratum=s, pop=pop, origin=origin,
                         piped=P, subcat_sum=K, gap=gap, gap_pop=gap * pop / 100,
                         n_nan_subcats=n_nan, ref_none=ref_none, no_who=no_who,
                         ref_piped=ref_piped, ref_nonpiped=ref_np,
                         ladder_resid=ladder_resid, subcat_short=subcat_short,
                         piped_matches_ref=(abs(P - ref_piped) < 0.51 if not pd.isna(ref_piped) else False)))

d = pd.DataFrame(rows)
tot = d.gap_pop.sum()
print(f"total gap pop: {tot/1e6:.1f}M over {len(d)} strata; strata with gap>0: {(d.gap>0.01).sum()}")

def bucket(row):
    if row.origin == "none":
        return "1_no_data_at_all"
    if row.origin == "missing_from_unified":
        return "1b_missing_from_unified"
    if row.n_nan_subcats == 4:
        return "2_stratum_all_nan(fillna0)"
    if row.ref_none and row.origin == "jmp":
        return "3_parser_ref_none(unnormalised)"
    if row.origin == "research":
        return "4_research_row_gap"
    return "5_jmp_incomplete_subcats"

d["bucket"] = d.apply(bucket, axis=1)
g = d.groupby("bucket").agg(gap_pop_M=("gap_pop", lambda x: x.sum()/1e6),
                            n_strata=("gap_pop", "size"),
                            n_with_gap=("gap", lambda x: (x > 0.01).sum()))
print(g.round(1).to_string())

# For bucket 5 (the honest JMP gap), split ladder residual vs subcategory shortfall
b5 = d[(d.bucket == "5_jmp_incomplete_subcats") & (d.gap > 0.01)].copy()
b5["ladder_pop"] = b5.ladder_resid.clip(lower=0) * b5["pop"] / 100
b5["short_pop"] = b5.subcat_short.clip(lower=0) * b5["pop"] / 100
print("\nbucket5 gap %.1fM; of which WHO-ladder residual %.1fM, subcat shortfall %.1fM (piped matches ref in %d/%d strata)" %
      (b5.gap_pop.sum()/1e6, b5.ladder_pop.sum()/1e6, b5.short_pop.sum()/1e6,
       b5.piped_matches_ref.sum(), len(b5)))

# top contributors per bucket
for b in sorted(d.bucket.unique()):
    sub = d[d.bucket == b].nlargest(6, "gap_pop")[["alpha3", "stratum", "gap", "gap_pop", "piped", "subcat_sum", "ref_piped", "ref_nonpiped"]]
    sub = sub[sub.gap_pop > 1e5]
    if len(sub):
        sub["gap_pop"] = (sub.gap_pop/1e6).round(2)
        print(f"\n== {b}"); print(sub.round(2).to_string(index=False))

SCRATCH = pathlib.Path(__file__).resolve().parent / "out"
SCRATCH.mkdir(exist_ok=True)
d.to_csv(SCRATCH / "gap_strata.csv", index=False)
