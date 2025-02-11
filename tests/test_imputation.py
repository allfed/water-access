import pandas as pd
import pytest
from scripts.Data_Manipulation_Scripts.imputation import test_append_country_info


def test_append_country_info():
    # Create a sample dataframe
    df = pd.DataFrame(
        {
            "alpha2": ["US", "CA", "GB"],
            "Country": ["United States", "Canada", "United Kingdom"],
        }
    )

    # Call the function to append country info
    df_output = append_country_info(df, "alpha2")

    # Check if the columns 'borders', 'region', and 'subregion' are added
    assert "borders" in df_output.columns
    assert "region" in df_output.columns
    assert "subregion" in df_output.columns

    # Check if the values in the new columns are correct
    assert df_output.loc[0, "borders"] == "CAN,MEX"
    assert df_output.loc[0, "region"] == "Americas"
    assert df_output.loc[0, "subregion"] == "Northern America"

    assert df_output.loc[1, "borders"] == "USA"
    assert df_output.loc[1, "region"] == "Americas"
    assert df_output.loc[1, "subregion"] == "Northern America"

    assert df_output.loc[2, "borders"] == ""
    assert df_output.loc[2, "region"] == "Europe"
    assert df_output.loc[2, "subregion"] == "Northern Europe"

    # Add more assertions if needed


# Run the tests
pytest.main()
