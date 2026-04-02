import polars as pl

# Define the path to your parquet files
parquet_path = "lemat_parquet_files/unique_pbe/*.parquet"

print("Scanning parquet files (Lazy Mode)...")
q = pl.scan_parquet(parquet_path)

# =====================================================================
# 1. GENERAL STATISTICS
# =====================================================================
print("\n" + "="*50)
print("GENERAL STATISTICS (Appropriate Numeric Columns)")
print("="*50)

# We explicitly select scalar numeric columns to avoid memory blowouts from large arrays
scalar_numeric_cols = [
    "nsites", 
    "nelements", 
    "energy", 
    "total_magnetization", 
    "dos_ef"
]

# Calculate summary statistics (count, null_count, mean, std, min, max, median)
# We use .collect() here to execute the lazy query and pull it into memory
stats_df = q.select(scalar_numeric_cols).collect().describe()

# Increase terminal output width so the table doesn't wrap awkwardly
with pl.Config(tbl_cols=10, tbl_width_chars=120):
    print(stats_df)


# =====================================================================
# 2. INVESTIGATE BY ANONYMOUS FORMULA
# =====================================================================
print("\n" + "="*50)
print("TOP 20 MOST COMMON ANONYMOUS FORMULAS")
print("="*50)

# Group by the anonymous formula, count them, and get average energy/sites
formula_query = (
    q.group_by("chemical_formula_anonymous")
     .agg([
         pl.len().alias("total_structures"),
         pl.col("energy").mean().alias("avg_energy"),
         pl.col("nsites").mean().alias("avg_nsites")
     ])
     .sort("total_structures", descending=True)
     .limit(20)
)

top_formulas_df = formula_query.collect()
print(top_formulas_df)


# =====================================================================
# 3. DEEP DIVE: LOWEST ENERGY STRUCTURES FOR A SPECIFIC FORMULA
# =====================================================================
# Let's dynamically grab the most common anonymous formula (e.g., "AB", "AB2", "A2B3")
most_common_formula = top_formulas_df[0, "chemical_formula_anonymous"]

print("\n" + "="*50)
print(f"DEEP DIVE: Top 10 Lowest Energy '{most_common_formula}' Structures")
print("="*50)

deep_dive_query = (
    q.filter(pl.col("chemical_formula_anonymous") == most_common_formula)
     .select([
         "immutable_id", 
         "chemical_formula_reduced", 
         "elements", 
         "energy",
         "functional"
     ])
     .sort("energy") # Sort by lowest energy
     .limit(10)
)

print(deep_dive_query.collect())
