import math

# 30% safety buffer on top of peak usage
CPU_BUFFER_FACTOR = 0.30
MEM_BUFFER_FACTOR = 0.30

# Target: peak usage must be >= 60% of the limit we set
TARGET_UTILIZATION = 0.60

# Round memory up to nearest 256 MiB block
MEM_ROUND_MIB = 256


def calculate_optimal_cpu(max_cpu_cores: float) -> dict:
    """
    Suggests the minimum integer CPU core count such that:
      - peak usage + 30% buffer fits within the limit
      - peak usage is >= 60% of the limit (quality gate)

    Returns a dict with all relevant figures for display.
    """
    buffered = max_cpu_cores * (1 + CPU_BUFFER_FACTOR)
    optimal_cores = max(1, math.ceil(buffered))

    # After integer rounding the utilisation ratio may differ from 1/(1+buffer)
    utilisation_pct = (max_cpu_cores / optimal_cores) * 100

    # Annotate whether the result clears the 60 % threshold
    # (for very small floats that round to 1, utilisation can drop below 60 %;
    #  1 is still the correct minimum so we keep it and flag it)
    meets_threshold = utilisation_pct >= (TARGET_UTILIZATION * 100)

    return {
        "max_usage_cores": round(max_cpu_cores, 3),
        "buffered_cores": round(buffered, 3),
        "optimal_cores": optimal_cores,
        "utilisation_pct": round(utilisation_pct, 1),
        "meets_60pct_threshold": meets_threshold,
        "buffer_applied_pct": int(CPU_BUFFER_FACTOR * 100),
    }


def calculate_optimal_memory(max_mem_bytes: float) -> dict:
    """
    Suggests the optimal memory limit in MiB, rounded up to the nearest
    256 MiB block, with a 30% buffer on top of peak usage.
    """
    max_mem_mib = max_mem_bytes / (1024 * 1024)
    buffered_mib = max_mem_mib * (1 + MEM_BUFFER_FACTOR)
    optimal_mib = max(MEM_ROUND_MIB, math.ceil(buffered_mib / MEM_ROUND_MIB) * MEM_ROUND_MIB)

    utilisation_pct = (max_mem_mib / optimal_mib) * 100
    meets_threshold = utilisation_pct >= (TARGET_UTILIZATION * 100)

    return {
        "max_usage_mib": round(max_mem_mib, 2),
        "max_usage_gib": round(max_mem_mib / 1024, 3),
        "buffered_mib": round(buffered_mib, 2),
        "optimal_mib": optimal_mib,
        "optimal_gib": round(optimal_mib / 1024, 3),
        "utilisation_pct": round(utilisation_pct, 1),
        "meets_60pct_threshold": meets_threshold,
        "buffer_applied_pct": int(MEM_BUFFER_FACTOR * 100),
    }


def compute_savings(
    current_cpu_limit_cores: float | None,
    optimal_cpu_cores: int,
    current_mem_limit_bytes: float | None,
    optimal_mem_mib: int,
) -> dict:
    """Compute absolute and percentage resource savings."""
    savings: dict = {}

    if current_cpu_limit_cores and current_cpu_limit_cores > 0:
        cpu_saved = current_cpu_limit_cores - optimal_cpu_cores
        savings["cpu_current_cores"] = round(current_cpu_limit_cores, 2)
        savings["cpu_optimal_cores"] = optimal_cpu_cores
        savings["cpu_saved_cores"] = round(cpu_saved, 2)
        savings["cpu_reduction_pct"] = round((cpu_saved / current_cpu_limit_cores) * 100, 1)
    else:
        savings["cpu_current_cores"] = None
        savings["cpu_optimal_cores"] = optimal_cpu_cores
        savings["cpu_saved_cores"] = None
        savings["cpu_reduction_pct"] = None

    if current_mem_limit_bytes and current_mem_limit_bytes > 0:
        current_mem_mib = current_mem_limit_bytes / (1024 * 1024)
        mem_saved_mib = current_mem_mib - optimal_mem_mib
        savings["mem_current_mib"] = round(current_mem_mib, 2)
        savings["mem_current_gib"] = round(current_mem_mib / 1024, 3)
        savings["mem_optimal_mib"] = optimal_mem_mib
        savings["mem_optimal_gib"] = round(optimal_mem_mib / 1024, 3)
        savings["mem_saved_mib"] = round(mem_saved_mib, 2)
        savings["mem_saved_gib"] = round(mem_saved_mib / 1024, 3)
        savings["mem_reduction_pct"] = round((mem_saved_mib / current_mem_mib) * 100, 1)
    else:
        savings["mem_current_mib"] = None
        savings["mem_current_gib"] = None
        savings["mem_optimal_mib"] = optimal_mem_mib
        savings["mem_optimal_gib"] = round(optimal_mem_mib / 1024, 3)
        savings["mem_saved_mib"] = None
        savings["mem_saved_gib"] = None
        savings["mem_reduction_pct"] = None

    return savings
