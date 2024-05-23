import pandas as pd
import pycountry
import pytest

def test_add_alpha_codes_from_ISO():
    # Create a sample dataframe
    df = pd.DataFrame({'Country': ['USA', 'CAN', 'MEX']})

    # Test when valid alpha3 codes are provided
    result = add_alpha_codes_from_ISO(df, 'Country')
    assert 'alpha2' in result.columns
    assert result['alpha2'].tolist() == ['US', 'CA', 'MX']

    # Test when an invalid alpha3 code is provided
    df = pd.DataFrame({'Country': ['USA', 'CAN', 'XYZ']})
    result = add_alpha_codes_from_ISO(df, 'Country')
    assert 'alpha2' in result.columns
    assert result['alpha2'].tolist() == ['US', 'CA', 'unk_XYZ']

    # Test when an empty dataframe is provided
    df = pd.DataFrame({'Country': []})
    result = add_alpha_codes_from_ISO(df, 'Country')
    assert 'alpha2' in result.columns
    assert result['alpha2'].tolist() == []

    # Test when the column name does not exist in the dataframe
    df = pd.DataFrame({'CountryName': ['USA', 'CAN', 'MEX']})
    with pytest.raises(KeyError):
        add_alpha_codes_from_ISO(df, 'Country')