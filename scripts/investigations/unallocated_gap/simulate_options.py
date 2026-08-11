"""Simulate gap-allocation options: effect on grid-affected population, global + focus countries."""
import pandas as pd
import numpy as np

SCRATCH = str(__import__("pathlib").Path(__file__).resolve().parent / "out")
__import__("pathlib").Path(SCRATCH).mkdir(exist_ok=True)
d = pd.read_csv(f"{SCRATCH}/gap_strata_split.csv")  # strata with gap>0.01 incl unimp/surface split
full = pd.read_csv(f"{SCRATCH}/gap_strata.csv")

REPO = "/Users/kevin/Documents/ProgrammingIsFun/ALLFED/water-access"
md = pd.read_csv(f"{REPO}/data/processed/merged_data.csv")
cats = ["Piped", "Packaged", "Delivered", "Borehole", "OtherUnpiped"]
md["urb_frac"] = md["% urban"] / 100

F = {"URBAN": 0.716, "RURAL": 0.287}  # HEMP midpoints

rows = []
for _, r in md.iterrows():
    for s, frac in (("URBAN", r["urb_frac"]), ("RURAL", 1 - r["urb_frac"])):
        pop = r["Population"] * frac
        v = {c: r[f"{s}{c}"] for c in cats}
        known = sum(v.values())
        gap = max(0.0, 100 - known)
        f = F[s]
        aff = v["Piped"] + v["Packaged"] + v["Delivered"] + f * v["Borehole"]
        rows.append(dict(alpha3=r["alpha3"], stratum=s, pop=pop, known=known, gap=gap,
                         aff=aff, **{c: v[c] for c in cats}))
sim = pd.DataFrame(rows)

split = d.set_index(["alpha3", "stratum"])[["g_other", "g_unknown"]]
sim = sim.join(split, on=["alpha3", "stratum"])
sim[["g_other", "g_unknown"]] = sim[["g_other", "g_unknown"]].fillna(0)

def affected_delta(row, mode):
    gap, known, aff = row.gap, row.known, row.aff
    if gap <= 0.01:
        return 0.0
    f = F[row.stratum]
    nonpiped_known = row.Packaged + row.Delivered + row.Borehole + row.OtherUnpiped
    nonpiped_aff = row.Packaged + row.Delivered + f * row.Borehole
    if mode == "b_prorata_all":
        return gap * (aff / known) if known > 0 else 0.0
    if mode == "e_all_vulnerable":
        return gap
    if mode == "f1_ladder_prorata_all":
        return row.g_unknown * (aff / known) if known > 0 else 0.0
    if mode == "f2_ladder_prorata_nonpiped":
        if nonpiped_known > 0:
            return row.g_unknown * (nonpiped_aff / nonpiped_known)
        return row.g_unknown * (aff / known) if known > 0 else 0.0
    return 0.0

for mode in ["b_prorata_all", "e_all_vulnerable", "f1_ladder_prorata_all", "f2_ladder_prorata_nonpiped"]:
    sim[mode] = sim.apply(lambda r: affected_delta(r, mode) * r["pop"] / 100, axis=1)

print("Baseline affected population: %.2fB" % ((sim.aff * sim["pop"] / 100).sum() / 1e9))
print("Gap population (status quo -> resilient): %.1fM\n" % ((sim.gap * sim["pop"] / 100).sum() / 1e6))
for mode in ["b_prorata_all", "e_all_vulnerable", "f1_ladder_prorata_all", "f2_ladder_prorata_nonpiped"]:
    tot = sim[mode].sum()
    print("%-28s global affected +%.1fM (+%.2f%% of baseline)" %
          (mode, tot / 1e6, 100 * tot / (sim.aff * sim["pop"] / 100).sum()))

focus = ["IND", "CHN", "PAK", "PHL", "MMR", "NER", "TCD", "SSD", "CAF", "NRU", "UKR"]
agg = sim[sim.alpha3.isin(focus)].groupby("alpha3").agg(
    gap_M=("gap", lambda x: (x * sim.loc[x.index, "pop"] / 100).sum() / 1e6),
    b_M=("b_prorata_all", lambda x: x.sum() / 1e6),
    e_M=("e_all_vulnerable", lambda x: x.sum() / 1e6),
    f1_M=("f1_ladder_prorata_all", lambda x: x.sum() / 1e6),
    f2_M=("f2_ladder_prorata_nonpiped", lambda x: x.sum() / 1e6),
).reindex(focus)
print("\nPer-country affected DELTA (millions) by option:")
print(agg.round(2).to_string())
sim.to_csv(f"{SCRATCH}/option_sim.csv", index=False)
