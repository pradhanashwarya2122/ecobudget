from carbon import estimate_co2e


def test_zero_bytes_yields_zero_emissions():
    result = estimate_co2e(0)
    assert result["kwh"] == 0
    assert result["estimated_co2e_grams"] == 0


def test_green_host_emits_less_than_grid_host():
    grid = estimate_co2e(1_000_000, green_host=False)
    green = estimate_co2e(1_000_000, green_host=True)
    assert green["estimated_co2e_grams"] < grid["estimated_co2e_grams"]


def test_emissions_scale_linearly_with_bytes():
    small = estimate_co2e(1_000)
    large = estimate_co2e(10_000)
    assert large["kwh"] == small["kwh"] * 10


def test_result_reports_input_bytes_and_host_flag():
    result = estimate_co2e(2048, green_host=True)
    assert result["bytes"] == 2048
    assert result["green_host"] is True
