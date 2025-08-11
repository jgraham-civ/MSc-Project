#!/bin/bash

### ESM1B ###

# esm-extract esm1b_t33_650M_UR50S ${HOME}/pipeline/datasets/ref_tier4_cdhit_90 ${HOME}/pipeline/esm/ref_tier4_cdhit_90_esm1b_embeddings --include mean

# esm-extract esm1b_t33_650M_UR50S ${HOME}/pipeline/datasets/tryp_combined_cleaned.fasta ${HOME}/pipeline/esm/tryp_combined_cleaned_esm1b_embeddings --include mean

# esm-extract esm1b_t33_650M_UR50S ${HOME}/pipeline/datasets/ref_tier1_cdhit_90 ${HOME}/pipeline/esm/ref_tier1_cdhit_90_esm1b_embeddings --include mean

# esm-extract esm1b_t33_650M_UR50S ${HOME}/pipeline/datasets/ref_tier2_cdhit_90 ${HOME}/pipeline/esm/ref_tier2_cdhit_90_esm1b_embeddings --include mean

# esm-extract esm1b_t33_650M_UR50S ${HOME}/pipeline/datasets/ref_tier3_cdhit_90 ${HOME}/pipeline/esm/ref_tier3_cdhit_90_esm1b_embeddings --include mean

### ESM2 ###

# esm-extract esm2_t33_650M_UR50D ${HOME}/pipeline/datasets/tryp_combined_cleaned.fasta ${HOME}/pipeline/esm/tryp_combined_cleaned_esm2_embeddings --include mean

esm-extract esm2_t33_650M_UR50D ${HOME}/pipeline/datasets/ref_tier1_cdhit_90 ${HOME}/pipeline/esm/ref_tier1_cdhit_90_esm2_embeddings --include mean

esm-extract esm2_t33_650M_UR50D ${HOME}/pipeline/datasets/ref_tier2_cdhit_90 ${HOME}/pipeline/esm/ref_tier2_cdhit_90_esm2_embeddings --include mean

esm-extract esm2_t33_650M_UR50D ${HOME}/pipeline/datasets/ref_tier3_cdhit_90 ${HOME}/pipeline/esm/ref_tier3_cdhit_90_esm2_embeddings --include mean

esm-extract esm2_t33_650M_UR50D ${HOME}/pipeline/datasets/ref_tier4_cdhit_90 ${HOME}/pipeline/esm/ref_tier4_cdhit_90_esm2_embeddings --include mean