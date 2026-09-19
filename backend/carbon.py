_KWH_PER_BYTE = 0.81e-9
_GRID_INTENSITY = 494
_GREEN_INTENSITY = 50


def estimate_co2e(total_bytes, green_host=False):
    kwh = total_bytes * _KWH_PER_BYTE
    intensity = _GREEN_INTENSITY if green_host else _GRID_INTENSITY
    grams = kwh * intensity
    return {
        "bytes": total_bytes,
        "kwh": kwh,
        "green_host": green_host,
        "estimated_co2e_grams": round(grams, 6),
        "note": "Estimated via SWD methodology, not measured emissions.",
    }