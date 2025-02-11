import pandas as pd


def handle_japan_case(row):
    """
    Handles missing data for Japan by assuming 100% URBANPiped and
    calculating RURALPiped based on the TOTALPiped and % urban.
    """
    if (
        row["Country"] == "Japan"
        and pd.isna(row["URBANPiped"])
        and pd.isna(row["RURALPiped"])
    ):
        row["URBANPiped"] = 100.0
        row["RURALPiped"] = (
            row["TOTALPiped"] - (row["% urban"] * row["URBANPiped"] / 100)
        ) / ((100 - row["% urban"]) / 100)
    return row


def process_water_data(df):
    """
    Processes water data by calculating TOTALPiped, filling missing values, and adding necessary columns.
    Ensures that missing values are filled from previous years' data.
    """

    # Replace '>99' with 100, '<1' with 0, and '-' with NaN
    df.replace({"<1": 0, ">99": 100, "-": pd.NA}, inplace=True)

    # Convert relevant columns to numeric for proper calculations
    columns_to_convert = [
        "TOTALPiped",
        "RURALPiped",
        "URBANPiped",
        "Population \r\n(thousands)",
        "% urban",
    ]
    df[columns_to_convert] = df[columns_to_convert].replace({" ": ""}, regex=True)
    df[columns_to_convert] = df[columns_to_convert].apply(
        pd.to_numeric, errors="coerce"
    )

    # Handle special cases:
    # If % urban is 0, set URBANPiped equal to RURALPiped
    df.loc[df["% urban"] == 0, "URBANPiped"] = df["RURALPiped"]

    # If % urban is 100 and RURALPiped is NaN, set RURALPiped equal to URBANPiped
    df.loc[(df["% urban"] == 100) & (df["RURALPiped"].isna()), "RURALPiped"] = df[
        "URBANPiped"
    ]

    # Handle Japan's special case
    df = df.apply(handle_japan_case, axis=1)

    # Fill missing rural and urban piped data where total piped is 100
    df.loc[(df["TOTALPiped"] == 100) & (df["RURALPiped"].isna()), "RURALPiped"] = 100
    df.loc[(df["TOTALPiped"] == 100) & (df["URBANPiped"].isna()), "URBANPiped"] = 100

    # Sort by country and year
    df = df.sort_values(["Country", "Year"])

    # Iterate through each country to ensure URBANPiped, RURALPiped, and TOTALPiped have values
    def fill_from_older_data(group):
        group["RURALPiped"] = (
            group["RURALPiped"].ffill().bfill()
        )  # Fill forward first, then backward
        group["URBANPiped"] = group["URBANPiped"].ffill().bfill()
        group["TOTALPiped"] = group["TOTALPiped"].ffill().bfill()
        return group

    # Apply the fill_from_older_data function to each country group
    df = (
        df.groupby("Country", group_keys=False)
        .apply(fill_from_older_data)
        .reset_index(drop=True)
    )

    # Recalculate TOTALPiped for all years
    df["TOTALPiped_Recalculated"] = (df["% urban"] / 100) * df["URBANPiped"] + (
        (100 - df["% urban"]) / 100
    ) * df["RURALPiped"]

    # Filter to keep only the most recent non-empty data entry per country
    df = df.dropna(subset=["URBANPiped", "RURALPiped", "TOTALPiped"], how="all")
    df = df.sort_values(["Country", "Year"], ascending=[True, False])
    df = df.drop_duplicates("Country", keep="first").reset_index(drop=True)

    return df


"""
Main function to process the input data file, interpolate missing data, and save the results.
"""

water_JMP_file_path = "/Users/kevin/Documents/ProgrammingIsFun/ALLFED/Water/water-access-gis/water-access/data/original_data/WHO Household Water Data - 2023 Data.csv"


# Read input data
df_water = pd.read_csv(water_JMP_file_path)

# Process water data
df = process_water_data(df_water)


# Count missing data points in each column
missing_data_counts = (
    df[["URBANPiped", "RURALPiped", "% urban", "TOTALPiped"]].isna().sum()
)

print("Missing Data Counts:")
print(missing_data_counts)

# Check correlation of missing data
missing_correlation = (
    df[["URBANPiped", "RURALPiped", "% urban", "TOTALPiped"]].isna().corr()
)

print("\nCorrelation of Missing Data:")
print(missing_correlation)


# Filter where TOTALPiped is present but URBANPiped and RURALPiped are missing
total_present_urban_rural_missing = df[
    df["TOTALPiped"].notna() & df["URBANPiped"].isna() & df["RURALPiped"].isna()
]

# Filter where URBANPiped is present but RURALPiped is missing
urban_present_rural_missing = df[df["URBANPiped"].notna() & df["RURALPiped"].isna()]

# Filter where RURALPiped is present but URBANPiped is missing
rural_present_urban_missing = df[df["RURALPiped"].notna() & df["URBANPiped"].isna()]

# Filter where URBANPiped or RURALPiped is present but TOTALPiped is missing
urban_or_rural_present_total_missing = df[
    (df["URBANPiped"].notna() | df["RURALPiped"].notna()) & df["TOTALPiped"].isna()
]

# Concatenate the results for all interesting cases
interesting_cases = pd.concat(
    [
        total_present_urban_rural_missing,
        urban_present_rural_missing,
        rural_present_urban_missing,
        urban_or_rural_present_total_missing,
    ]
).drop_duplicates()

# Select only the relevant columns
interesting_columns = [
    "Country",
    "Year",
    "URBANPiped",
    "RURALPiped",
    "% urban",
    "TOTALPiped",
]
interesting_cases = interesting_cases[interesting_columns]


display(interesting_cases)

# also display japan
display(df[df["Country"] == "Japan"][interesting_columns])
