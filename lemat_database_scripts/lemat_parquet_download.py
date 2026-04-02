from huggingface_hub import snapshot_download

# Define where you want the files saved on your computer
save_directory = "./lemat_parquet_files"

print("Starting download...")

# Download only the parquet files
snapshot_download(
    repo_id="LeMaterial/LeMat-BulkUnique",
    repo_type="dataset",
    local_dir=save_directory,
    # This filter ensures you only download the parquet files for the 'unique_pbe' subset
    # If the files are in the root directory, use "*.parquet" instead
    allow_patterns="*unique_pbe/*.parquet" 
)

print(f"Download complete! Files are saved in: {save_directory}")
