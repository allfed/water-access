"""Split the JMP-origin gap into unimproved+surface (genuine other-unpiped)
vs improved-non-piped shortfall (truly unknown), using WHO same-vintage columns."""
import pandas as pd
import numpy as np

SCRATCH = str(__import__("pathlib").Path(__file__).resolve().parent / "out")
__import__("pathlib").Path(SCRATCH).mkdir(exist_ok=True)
REPO = "/Users/kevin/Documents/ProgrammingIsFun/ALLFED/water-access"

d = pd.read_csv(f"{SCRATCH}/gap_strata.csv")
who = pd.read_csv(f"{REPO}/data/original_data/WHO Household Water Data - 2023 Data.csv", low_memory=False)
who = who.replace({"<1": 0.5, ">99": 100, "-": pd.NA})
for s in ["URBAN", "RURAL"]:
    for c in ["Piped", "Non-piped", "Unimproved", "Surface water"]:
        who[f"{s}{c}"] = pd.to_numeric(who[f"{s}{c}"], errors="coerce")
who["Year"] = pd.to_numeric(who["Year"], errors="coerce")

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
latest = who.sort_values(["Country", "Year"], ascending=[True, False]).drop_duplicates("Country", keep="first")
latest = latest.drop_duplicates("alpha3", keep="first").set_index("alpha3")

# sanity: ladder identity
for s in ["URBAN", "RURAL"]:
    tot = latest[[f"{s}Piped", f"{s}Non-piped", f"{s}Unimproved", f"{s}Surface water"]].sum(axis=1)
    ok = tot.dropna()
    print(f"{s} ladder identity: {len(ok)} strata, sum mean {ok.mean():.1f}, within [98,102]: {((ok>=98)&(ok<=102)).sum()}")

def get(a3, col):
    return latest.loc[a3, col] if a3 in latest.index else np.nan

d["unimp"] = d.apply(lambda r: get(r.alpha3, f"{r.stratum}Unimproved"), axis=1)
d["surface"] = d.apply(lambda r: get(r.alpha3, f"{r.stratum}Surface water"), axis=1)

b5 = d[(d.bucket == "5_jmp_incomplete_subcats") & (d.gap > 0.01)].copy()
b5["g_other"] = np.minimum(b5.gap, (b5.unimp.fillna(0) + b5.surface.fillna(0)))  # genuinely other-unpiped
b5["g_unknown"] = (b5.gap - b5.g_other).clip(lower=0)                            # improved non-piped shortfall
b5["other_pop"] = b5.g_other * b5["pop"] / 100
b5["unknown_pop"] = b5.g_unknown * b5["pop"] / 100
print("\nbucket5 gap %.1fM = genuinely-other (unimproved+surface) %.1fM + unknown improved-non-piped %.1fM"
      % (b5.gap_pop.sum()/1e6, b5.other_pop.sum()/1e6, b5.unknown_pop.sum()/1e6))

top = b5.groupby("alpha3").agg(gap_M=("gap_pop", lambda x: x.sum()/1e6),
                               other_M=("other_pop", lambda x: x.sum()/1e6),
                               unknown_M=("unknown_pop", lambda x: x.sum()/1e6)).nlargest(15, "gap_M")
print("\nper-country split (top 15):")
print(top.round(1).to_string())

# same split for ALL buckets combined (whole 348M)
d2 = d[d.gap > 0.01].copy()
d2["g_other"] = np.minimum(d2.gap, (d2.unimp.fillna(0) + d2.surface.fillna(0)))
d2["g_unknown"] = (d2.gap - d2.g_other).clip(lower=0)
print("\nALL buckets: gap %.1fM = other %.1fM + unknown %.1fM" %
      ((d2.gap*d2["pop"]/100).sum()/1e6, (d2.g_other*d2["pop"]/100).sum()/1e6, (d2.g_unknown*d2["pop"]/100).sum()/1e6))

# relative worst countries split
worst = d2[d2.alpha3.isin(["NRU","CAF","TCD","SSD","MMR","NER","PHL","PAK","IND","CHN"])]
w = worst.groupby("alpha3").agg(gap_M=("gap_pop", lambda x: x.sum()/1e6),
    other_M=("g_other", lambda x: (x*worst.loc[x.index,"pop"]/100).sum()/1e6),
    unknown_M=("g_unknown", lambda x: (x*worst.loc[x.index,"pop"]/100).sum()/1e6))
print("\nfocus countries:"); print(w.round(2).to_string())
d2.to_csv(f"{SCRATCH}/gap_strata_split.csv", index=False)
