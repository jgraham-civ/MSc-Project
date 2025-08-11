#!/bin/sh

#### SCRIPT FOR REMOVING REDUNDANT SEQUENCES FROM REFERENCE SETS ####

### --- Tier 1 --- ###

# Threshold 0.9

cd-hit -i ${HOME}/pipeline/datasets/ref_tier1.fasta -o ref_tier1_cdhit_90 -c 0.90 -n 5 -d 0

# Threshold 0.7

cd-hit -i ${HOME}/pipeline/datasets/ref_tier1.fasta -o ref_tier1_cdhit_70 -c 0.70 -n 5 -d 0

# Threshold 0.5

cd-hit -i ${HOME}/pipeline/datasets/ref_tier1.fasta -o ref_tier1_cdhit_50 -c 0.50 -n 3 -d 0


### --- Tier 2 --- ###

# Threshold 0.9

cd-hit -i ${HOME}/pipeline/datasets/ref_tier2.fasta -o ref_tier2_cdhit_90 -c 0.90 -n 5 -d 0

# Threshold 0.7

cd-hit -i ${HOME}/pipeline/datasets/ref_tier2.fasta -o ref_tier2_cdhit_70 -c 0.70 -n 5 -d 0

# Threshold 0.5

cd-hit -i ${HOME}/pipeline/datasets/ref_tier2.fasta -o ref_tier2_cdhit_50 -c 0.50 -n 3 -d 0


### --- Tier 3 --- ###

# Threshold 0.9

cd-hit -i ${HOME}/pipeline/datasets/ref_tier3.fasta -o ref_tier3_cdhit_90 -c 0.90 -n 5 -d 0

# Threshold 0.7

cd-hit -i ${HOME}/pipeline/datasets/ref_tier3.fasta -o ref_tier3_cdhit_70 -c 0.70 -n 5 -d 0

# Threshold 0.5

cd-hit -i ${HOME}/pipeline/datasets/ref_tier3.fasta -o ref_tier3_cdhit_50 -c 0.50 -n 3 -d 0


### --- Tier 4 --- ###

# Threshold 0.9

cd-hit -i ${HOME}/pipeline/datasets/ref_tier4.fasta -o ref_tier4_cdhit_90 -c 0.90 -n 5 -d 0

## Threshold 0.7

cd-hit -i ${HOME}/pipeline/datasets/ref_tier4.fasta -o ref_tier4_cdhit_70 -c 0.70 -n 5 -d 0

## Threshold 0.5

cd-hit -i ${HOME}/pipeline/datasets/ref_tier4.fasta -o ref_tier4_cdhit_50 -c 0.50 -n 3 -d 0