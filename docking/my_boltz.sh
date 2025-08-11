#!/bin/bash

files=("SAM_yamls/U_SAM.yaml" "SAM_yamls/V_SAM.yaml" "SAM_yamls/W_SAM.yaml" "SAM_yamls/X_SAM.yaml" "SAM_yamls/XX_SAM.yaml" "SAM_yamls/Y_SAM.yaml" "SAM_yamls/Z_SAM.yaml")

for file in "${files[@]}"; do
    boltz predict $file --use_msa_server
done