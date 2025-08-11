#!/bin/sh

#### SCRIPT FOR DOWNLOADING SEQUENCES ####

### -- REFERENCE SET -- ###

# Tier 1: N-terminal methyltransferases
curl "https://rest.uniprot.org/uniprotkb/stream?query=(go:0071885)+AND+(taxonomy_id:2759)+AND+(cc_catalytic_activity:%22CHEBI:15414%22)+AND+(reviewed:true)&format=fasta" -o ref_tier1.fasta

# Tier 2: Any histone H3 methyltransferase
curl "https://rest.uniprot.org/uniprotkb/stream?query=(go:0140938)+AND+(taxonomy_id:2759)+AND+(cc_catalytic_activity:%22CHEBI:15414%22)+AND+(reviewed:true)&format=fasta" -o ref_tier2.fasta

# Tier 3: Any histone methyltransferase
curl "https://rest.uniprot.org/uniprotkb/stream?query=(go:0042054)+AND+(taxonomy_id:2759)+AND+(cc_catalytic_activity:%22CHEBI:15414%22)+AND+(reviewed:true)&format=fasta" -o ref_tier3.fasta

# Tier 4: Any protein methyltransferase
curl "https://rest.uniprot.org/uniprotkb/stream?query=(go:0008276)+AND+(taxonomy_id:2759)+AND+(cc_catalytic_activity:%22CHEBI:15414%22)+AND+(reviewed:true)&format=fasta" -o ref_tier4.fasta

# Tier 5: Any methyltransferase
curl "https://rest.uniprot.org/uniprotkb/stream?query=(go:0008168)+AND+(taxonomy_id:2759)+AND+(cc_catalytic_activity:%22CHEBI:15414%22)+AND+(reviewed:true)&format=fasta" -o ref_tier5.fasta


### -- TRYPANOSOMA BRUCEI SET -- ###

# Search 1: UniProt
curl "https://rest.uniprot.org/uniprotkb/stream?query=(taxonomy_id:185431)&format=fasta" -o tryp_uniprot.fasta

# Search 2: NCBI
esearch -db protein -query "txid185431[Organism:exp]" | efetch -format fasta > tryp_ncbi.fasta

# Combine UniProt and NCBI results
cat tryp_uniprot.fasta tryp_ncbi.fasta > tryp_combined.fasta