import subprocess

# List of candidates
candidates = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "AA",
    "AB",
    "AC",
    "AD",
    "AE",
    "AF",
    "AG",
    "AH",
    "AI",
    "AJ",
    "AK",
    "AL",
    "AM",
    "AN",
    "AO",
    "AP",
    "AQ",
    "AR",
    "AS",
    "AT",
    "AU",
    "AV",
    "AW",
    "AX",
    "AY",
    "AZ",
    "BA",
    "BB",
    "BC",
    "BD",
    "BE",
    "BF",
    "BG",
    "BH",
    "BI",
    "XX",
]

# Re-Docking
for i in candidates:
    subprocess.run([
        "gnina",
        "-r", f"SAM_inputfiles/{i}_SAM_model_0_chainA.pdb",
        "-l", f"SAM_inputfiles/{i}_SAM_model_0_chainB.pdb",
        "--autobox_ligand", f"SAM_inputfiles/{i}_SAM_model_0_chainB.pdb",
        "-o", f"SAM_outputfiles/{i}_SAM_gnina.pdb", "--log", f"SAM_outputfiles/{i}_SAM_gnina.log"
    ])

# Score Only
for i in candidates:
    subprocess.run([
        "gnina",
        "-r", f"SAM_inputfiles/{i}_SAM_model_0_chainA.pdb",
        "-l", f"SAM_inputfiles/{i}_SAM_model_0_chainB.pdb",
        "--score_only", "--log", f"SAM_outputfiles/{i}_SAM_gnina_scoreonly.log"
    ])