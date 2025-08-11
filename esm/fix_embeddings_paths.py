import os
import shutil

def flatten_esm_pt_files(source_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for entry in os.listdir(source_dir):
        full_path = os.path.join(source_dir, entry)

        if os.path.isfile(full_path) and entry.endswith('.pt'):
            # Already valid .pt file, copy as-is
            dest = os.path.join(output_dir, entry)
            shutil.copy2(full_path, dest)
            print(f"Copied valid file: {full_path} → {dest}")

        elif os.path.isdir(full_path):
            pt_files = [f for f in os.listdir(full_path) if f.endswith('.pt')]

            if len(pt_files) == 1:
                pt_file = pt_files[0]
                old_file_path = os.path.join(full_path, pt_file)

                # Flatten by joining dir name and filename (no separators)
                new_file_name = f"{entry}{pt_file}"
                dest = os.path.join(output_dir, new_file_name)

                shutil.copy2(old_file_path, dest)
                print(f"Flattened and copied: {old_file_path} → {dest}")

            else:
                print(f"Skipped directory (not exactly 1 .pt file): {full_path}")


def flatten_esm_pt_files_recursive(source_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".pt"):
                full_path = os.path.join(root, file)
                
                # Get the relative path components from source_dir
                rel_path = os.path.relpath(full_path, source_dir)
                parts = rel_path.split(os.sep)  # splits all nested folders
                
                # Join all parts (folders + filename) into one flat filename, removing slashes
                # Optionally, remove spaces or other special chars here if needed
                flat_name = "".join(parts)
                
                dest_path = os.path.join(output_dir, flat_name)
                shutil.copy2(full_path, dest_path)
                print(f"Flattened: {full_path} → {dest_path}")

# ESM1b
flatten_esm_pt_files_recursive(os.path.expandvars("${HOME}/pipeline/esm/tryp_combined_cleaned_esm1b_embeddings"), os.path.expandvars("${HOME}/pipeline/esm/tryp_combined_cleaned_esm1b_embeddings_flattened"))
flatten_esm_pt_files_recursive(os.path.expandvars("${HOME}/pipeline/esm/ref_tier1_cdhit_90_esm1b_embeddings"), os.path.expandvars("${HOME}/pipeline/esm/ref_tier1_cdhit_90_esm1b_embeddings_flattened"))
flatten_esm_pt_files_recursive(os.path.expandvars("${HOME}/pipeline/esm/ref_tier2_cdhit_90_esm1b_embeddings"), os.path.expandvars("${HOME}/pipeline/esm/ref_tier2_cdhit_90_esm1b_embeddings_flattened"))
flatten_esm_pt_files_recursive(os.path.expandvars("${HOME}/pipeline/esm/ref_tier3_cdhit_90_esm1b_embeddings"), os.path.expandvars("${HOME}/pipeline/esm/ref_tier3_cdhit_90_esm1b_embeddings_flattened"))
flatten_esm_pt_files_recursive(os.path.expandvars("${HOME}/pipeline/esm/ref_tier4_cdhit_90_esm1b_embeddings"), os.path.expandvars("${HOME}/pipeline/esm/ref_tier4_cdhit_90_esm1b_embeddings_flattened"))

# ESM2
flatten_esm_pt_files_recursive(os.path.expandvars("${HOME}/pipeline/esm/tryp_combined_cleaned_esm2_embeddings"), os.path.expandvars("${HOME}/pipeline/esm/tryp_combined_cleaned_esm2_embeddings_flattened"))
flatten_esm_pt_files_recursive(os.path.expandvars("${HOME}/pipeline/esm/ref_tier1_cdhit_90_esm2_embeddings"), os.path.expandvars("${HOME}/pipeline/esm/ref_tier1_cdhit_90_esm2_embeddings_flattened"))
flatten_esm_pt_files_recursive(os.path.expandvars("${HOME}/pipeline/esm/ref_tier2_cdhit_90_esm2_embeddings"), os.path.expandvars("${HOME}/pipeline/esm/ref_tier2_cdhit_90_esm2_embeddings_flattened"))
flatten_esm_pt_files_recursive(os.path.expandvars("${HOME}/pipeline/esm/ref_tier3_cdhit_90_esm2_embeddings"), os.path.expandvars("${HOME}/pipeline/esm/ref_tier3_cdhit_90_esm2_embeddings_flattened"))
flatten_esm_pt_files_recursive(os.path.expandvars("${HOME}/pipeline/esm/ref_tier4_cdhit_90_esm2_embeddings"), os.path.expandvars("${HOME}/pipeline/esm/ref_tier4_cdhit_90_esm2_embeddings_flattened"))