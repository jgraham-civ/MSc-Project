#!/bin/bash

### Count the number of sequences in each dataset

grep -c "^>" ${HOME}/pipeline/datasets/ref_tier1_cdhit_90

grep -c "^>" ${HOME}/pipeline/datasets/ref_tier2_cdhit_90

grep -c "^>" ${HOME}/pipeline/datasets/ref_tier3_cdhit_90

grep -c "^>" ${HOME}/pipeline/datasets/ref_tier4_cdhit_90

grep -c "^>" ${HOME}/pipeline/datasets/tryp_combined_cleaned.fasta