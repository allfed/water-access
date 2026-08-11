"""Phase 0 gating experiment for the 30 arc-second feasibility question.

Measures, per country and globally, the resolution-induced bias in
population-weighted distance-to-surface-water between:

  - dtw @ 30 arc-second (Kummu et al. native 1 km product), and
  - dtw @ 5 arc-minute broadcast back onto the 30as grid (each fine cell
    inherits its coarse cell's value — mathematically identical to the
    pop-weighted statistics the current 5 arc-minute pipeline computes).

Population weights are GHS-POP E2020 at 30 arc-second in both cases, so the
ONLY thing that differs between the two columns is the resolution of the
distance layer. Runs in latitude strips; peak RAM ~6 GB.

Outputs (results/phase0/):
  phase0_per_country.csv   per-ISO3 paired statistics
  phase0_summary.csv       global + continental rollup
  phase0_histograms.npz    per-country pop-weighted dtw histograms, both res
"""

import time

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.windows import Window, from_bounds

RASTER_DIR = "data/GIS/source_rasters"
DTW30 = f"{RASTER_DIR}/distance2water_30arcsec.tif"
DTW5 = f"{RASTER_DIR}/distance2water_5arcmin.tif"
POP30 = f"{RASTER_DIR}/ghs_pop_2020_30ss.tif"
ADM0 = f"{RASTER_DIR}/geoBoundariesCGAZ_ADM0.gpkg"
OUT_DIR = "results/phase0"

STRIP_ROWS = 2160  # 18 degrees of latitude per strip at 30as
# One-way walk thresholds (km). 6.5 km ~ the model's median walk limit at the
# 2-4 h gathering-time budget; the rest span the Monte Carlo range.
THRESHOLDS_KM = [1, 2, 3, 4, 5, 6, 6.5, 7, 8, 10, 15, 20]
HIST_EDGES_M = np.concatenate([np.arange(0, 30001, 250), [1e9]])  # 250 m bins


def build_country_raster(shape, transform):
    adm0 = gpd.read_file(ADM0)
    iso_field = "shapeGroup" if "shapeGroup" in adm0.columns else "shapeISO"
    adm0 = adm0[adm0[iso_field].notna()].reset_index(drop=True)
    isos = sorted(adm0[iso_field].unique())
    iso_index = {iso: i + 1 for i, iso in enumerate(isos)}  # 0 = unassigned
    shapes = (
        (geom, iso_index[iso])
        for geom, iso in zip(adm0.geometry, adm0[iso_field])
    )
    t0 = time.time()
    cid = features.rasterize(
        shapes, out_shape=shape, transform=transform, fill=0, dtype="int16"
    )
    print(f"country raster {shape} in {time.time()-t0:.0f}s", flush=True)
    return cid, isos


def read_aligned(src, bounds, out_shape, fill):
    """Boundless read of src over `bounds`, asserting exact grid alignment."""
    win = from_bounds(*bounds, transform=src.transform)
    # Sub-cell tolerance: fractional offsets get snapped (nearest-neighbour,
    # max ~0.5 cell shift) which affects both resolution columns identically.
    assert abs(win.height - out_shape[0]) < 0.01, (win, out_shape)
    assert abs(win.width - out_shape[1]) < 0.01, (win, out_shape)
    win = Window(round(win.col_off), round(win.row_off), out_shape[1], out_shape[0])
    return src.read(1, window=win, boundless=True, fill_value=fill)


def main():
    src30 = rasterio.open(DTW30)
    src5 = rasterio.open(DTW5)
    srcpop = rasterio.open(POP30)
    H, W = src30.height, src30.width
    nodata30 = src30.nodata
    nodata5 = src5.nodata
    print(
        f"dtw30 {W}x{H} nodata={nodata30} | dtw5 {src5.width}x{src5.height} "
        f"nodata={nodata5} | pop {srcpop.width}x{srcpop.height} "
        f"nodata={srcpop.nodata}",
        flush=True,
    )
    assert abs(src30.transform.a - src5.transform.a / 10) < 1e-9

    cid_full, isos = build_country_raster((H, W), src30.transform)
    n = len(isos) + 1
    nthr = len(THRESHOLDS_KM)
    nbins = len(HIST_EDGES_M) - 1

    pop_sum = np.zeros(n)          # pop on paired-valid cells
    popdtw30 = np.zeros(n)         # sum pop * dtw30
    popdtw5 = np.zeros(n)          # sum pop * dtw5(broadcast)
    thr30 = np.zeros((n, nthr))    # pop with dtw30 < thr
    thr5 = np.zeros((n, nthr))
    hist30 = np.zeros((n, nbins))
    hist5 = np.zeros((n, nbins))
    # population accounting across validity masks
    pop_total = np.zeros(n)        # all pop on the strip grid
    pop_only30 = np.zeros(n)       # dtw30 valid, dtw5 nodata (coastal gain)
    pop_only5 = np.zeros(n)        # dtw5 valid, dtw30 nodata
    pop_neither = np.zeros(n)      # pop on cells with no dtw at either res

    for row_off in range(0, H, STRIP_ROWS):
        h = min(STRIP_ROWS, H - row_off)
        t0 = time.time()
        win = Window(0, row_off, W, h)
        bounds = rasterio.windows.bounds(win, src30.transform)

        dtw30 = src30.read(1, window=win).astype("float64")
        v30 = np.isfinite(dtw30) & (dtw30 >= 0)
        if nodata30 is not None:
            v30 &= dtw30 != nodata30

        coarse = read_aligned(
            src5,
            bounds,
            (h // 10, W // 10),
            fill=nodata5 if nodata5 is not None else -1,
        ).astype("float64")
        dtw5 = np.repeat(np.repeat(coarse, 10, axis=0), 10, axis=1)
        v5 = np.isfinite(dtw5) & (dtw5 >= 0)
        if nodata5 is not None:
            v5 &= dtw5 != nodata5

        pop = read_aligned(srcpop, bounds, (h, W), fill=0).astype("float64")
        pop[~np.isfinite(pop) | (pop < 0)] = 0

        cid = cid_full[row_off : row_off + h]
        both = v30 & v5

        pop_total += np.bincount(cid.ravel(), weights=pop.ravel(), minlength=n)
        for mask, acc in [
            (v30 & ~v5, pop_only30),
            (v5 & ~v30, pop_only5),
            (~v30 & ~v5, pop_neither),
        ]:
            acc += np.bincount(
                cid[mask], weights=pop[mask], minlength=n
            )

        c, p = cid[both].astype(np.int64), pop[both]
        d30, d5 = dtw30[both], dtw5[both]
        pop_sum += np.bincount(c, weights=p, minlength=n)
        popdtw30 += np.bincount(c, weights=p * d30, minlength=n)
        popdtw5 += np.bincount(c, weights=p * d5, minlength=n)
        for j, thr in enumerate(THRESHOLDS_KM):
            m = thr * 1000.0
            thr30[:, j] += np.bincount(c[d30 < m], weights=p[d30 < m], minlength=n)
            thr5[:, j] += np.bincount(c[d5 < m], weights=p[d5 < m], minlength=n)
        b30 = np.clip(np.digitize(d30, HIST_EDGES_M) - 1, 0, nbins - 1)
        b5 = np.clip(np.digitize(d5, HIST_EDGES_M) - 1, 0, nbins - 1)
        hist30 += np.bincount(
            c * nbins + b30, weights=p, minlength=n * nbins
        ).reshape(n, nbins)
        hist5 += np.bincount(
            c * nbins + b5, weights=p, minlength=n * nbins
        ).reshape(n, nbins)
        print(
            f"strip rows {row_off}-{row_off+h}: {time.time()-t0:.1f}s "
            f"(paired pop {p.sum()/1e6:.1f} M)",
            flush=True,
        )

    def median_from_hist(hist):
        med = np.full(hist.shape[0], np.nan)
        centers = (HIST_EDGES_M[:-1] + np.minimum(HIST_EDGES_M[1:], 60000)) / 2
        for i in range(hist.shape[0]):
            tot = hist[i].sum()
            if tot <= 0:
                continue
            cum = np.cumsum(hist[i])
            med[i] = centers[np.searchsorted(cum, tot / 2)]
        return med

    labels = ["UNASSIGNED"] + isos
    df = pd.DataFrame(
        {
            "ISO3": labels,
            "pop_total_M": pop_total / 1e6,
            "pop_paired_M": pop_sum / 1e6,
            "pop_only30_M": pop_only30 / 1e6,
            "pop_only5_M": pop_only5 / 1e6,
            "pop_neither_M": pop_neither / 1e6,
            "pw_mean_dtw30_km": np.divide(
                popdtw30, pop_sum, out=np.full(n, np.nan), where=pop_sum > 0
            )
            / 1000,
            "pw_mean_dtw5_km": np.divide(
                popdtw5, pop_sum, out=np.full(n, np.nan), where=pop_sum > 0
            )
            / 1000,
            "pw_median_dtw30_km": median_from_hist(hist30) / 1000,
            "pw_median_dtw5_km": median_from_hist(hist5) / 1000,
        }
    )
    for j, thr in enumerate(THRESHOLDS_KM):
        with np.errstate(invalid="ignore"):
            df[f"pct_within_{thr}km_30as"] = np.divide(
                thr30[:, j], pop_sum, out=np.full(n, np.nan), where=pop_sum > 0
            ) * 100
            df[f"pct_within_{thr}km_5min"] = np.divide(
                thr5[:, j], pop_sum, out=np.full(n, np.nan), where=pop_sum > 0
            ) * 100
            df[f"delta_pp_{thr}km"] = (
                df[f"pct_within_{thr}km_30as"] - df[f"pct_within_{thr}km_5min"]
            )
    df["delta_mean_km"] = df.pw_mean_dtw30_km - df.pw_mean_dtw5_km
    df.to_csv(f"{OUT_DIR}/phase0_per_country.csv", index=False)
    np.savez_compressed(
        f"{OUT_DIR}/phase0_histograms.npz",
        hist30=hist30,
        hist5=hist5,
        edges_m=HIST_EDGES_M,
        labels=np.array(labels),
    )

    # Rollups: global and continental (continent map from the repo's own file)
    merged = pd.read_csv("data/processed/merged_data.csv")[
        ["alpha3", "Continent"]
    ].drop_duplicates()
    df = df.merge(merged, left_on="ISO3", right_on="alpha3", how="left")
    rows = []
    for name, sub in [("GLOBAL", df[df.ISO3 != "UNASSIGNED"])] + [
        (c, g) for c, g in df.groupby("Continent")
    ]:
        w = sub.pop_paired_M
        if w.sum() <= 0:
            continue
        row = {
            "region": name,
            "pop_paired_M": w.sum(),
            "pw_mean_dtw30_km": (sub.pw_mean_dtw30_km * w).sum() / w.sum(),
            "pw_mean_dtw5_km": (sub.pw_mean_dtw5_km * w).sum() / w.sum(),
        }
        for thr in THRESHOLDS_KM:
            row[f"pct_within_{thr}km_30as"] = (
                sub[f"pct_within_{thr}km_30as"] * w
            ).sum() / w.sum()
            row[f"pct_within_{thr}km_5min"] = (
                sub[f"pct_within_{thr}km_5min"] * w
            ).sum() / w.sum()
            row[f"delta_pp_{thr}km"] = (
                row[f"pct_within_{thr}km_30as"] - row[f"pct_within_{thr}km_5min"]
            )
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/phase0_summary.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
