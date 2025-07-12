import pandas as pd
from pathlib import Path


def load_data(file_path):
    """Load data from CSV file."""
    return pd.read_csv(file_path)


def preprocess_data(df):
    """Preprocess the dataframe."""
    df = df.drop(["shapeType"], axis=1)
    return df.dropna(subset=["shapeGroup"])


def create_nan_indicators(df, columns):
    """Create NaN indicator columns."""
    for col in columns:
        df[f"{col}_nan"] = df[col].isna().astype(int)
    df["any_nan"] = df[columns].isna().any(axis=1).astype(int)
    return df


def calculate_nan_correlation(df, columns):
    """Calculate correlation matrix for NaN presence."""
    nan_indicator = df[columns].isna().astype(int)
    return nan_indicator.corr()


def count_single_nan_columns(df, columns):
    """Count rows where only one column has NaN."""
    counts = {}
    for col in columns:
        other_cols = [c for c in columns if c != col]
        counts[col] = len(df[df[col].isna() & df[other_cols].notna().all(axis=1)])
    return counts


def drop_nan_rows(df, columns):
    """Drop rows with NaN in specified columns."""
    return df.dropna(subset=columns)


def rename_columns(df, column_mapping):
    """Rename columns based on mapping."""
    return df.rename(columns=column_mapping)


def save_data(df, file_path):
    """Save dataframe to CSV."""
    df.to_csv(file_path, index=False)


def main():
    input_file = Path("../../data/GIS/updated_GIS_output.csv")
    output_file = Path("../../data/GIS/GIS_pre_processed_for_analysis.csv")
    output_file_merge = Path("../../data/GIS/GIS_country_points_fid.csv")
    nan_columns = ["dtw_1", "pop_density", "GHS_SMOD", "slope_1"]

    df = load_data(input_file)
    df = preprocess_data(df)

    column_mapping = {
        "dtw": "dtw_1",
        "slope1": "slope_1",
        "shapeGroup": "ISOCODE",
        "tp51": "grip_1_1",
        "tp41": "grip_2_1",
        "tp31": "grip_3_1",
        "tp21": "grip_4_1",
        "tp11": "grip_5_1",
        "id": "fid",
    }
    df = rename_columns(df, column_mapping)

    df = create_nan_indicators(df, nan_columns)

    nan_correlation = calculate_nan_correlation(df, nan_columns)
    print("Correlation between NaN values in columns:")
    print(nan_correlation)

    single_nan_counts = count_single_nan_columns(df, nan_columns)
    for col, count in single_nan_counts.items():
        print(f"Number of rows where {col} is the only NaN value: {count}")

    pre_len = len(df)
    df = drop_nan_rows(df, nan_columns)
    print(f"Number of rows dropped: {pre_len - len(df)}")

    # export two files, one with the original columns and one with the new
    # columns
    # drop columns left, top, right, bottom for the first
    df_to_process = df.drop(columns=["left", "top", "right", "bottom"])
    # only keep fid left top right bottom for the second
    df_to_merge = df[["fid", "left", "top", "right", "bottom"]]

    save_data(df_to_process, output_file)
    save_data(df_to_merge, output_file_merge)
    print(f"Data saved to {output_file}")


if __name__ == "__main__":
    main()
