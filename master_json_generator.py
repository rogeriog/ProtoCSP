import os
import json
from itertools import combinations

# --- 1. PROTOTYPE DEFINITIONS & SITES ---
PROTOTYPES = {
    "A3B2X9": {
        "name": "Cs3Sb2I9_Vacancy_Ordered",
        "cif": "Cs3Sb2I9.cif",
        "sites": {"A": "Cs", "B": ["Sb"], "X": "I"}
    },
    "ABX3": {
        "name": "CsPbI3_Standard_3D",
        "cif": "CsPbI3.cif",
        "sites": {"A": "Cs", "B": ["Pb"], "X": "I"}
    },
    "A2B1B2X6": {
        "name": "Cs2AgBiI6_Double_Perovskite",
        "cif": "Cs2AgBiI6.cif",
        "sites": {"A": "Cs", "B": ["Ag", "Bi"], "X": "I"}
    },
    "A2BX6": {
        "name": "Cs2SnI6_Vacancy_Double",
        "cif": "Cs2SnI6.cif",
        "sites": {"A": "Cs", "B": ["Sn"], "X": "I"}
    }
}

# --- 2. POLYATOMIC DUMMY MAPPING ---
POLYATOMIC_MAPPING = {
    "Fr": {"name": "NH4", "center": "N"},
    "At": {"name": "CN",  "center": "C"},
    "Po": {"name": "SCN", "center": "S"}
}

def apply_dummies(element_set):
    mapping = {"NH4": "Fr", "CN": "At", "SCN": "Po"}
    return {mapping.get(e, e) for e in element_set}

# --- 3. CHEMICAL SUBSTITUTION LISTS ---
# A and X sites merge High and Restr into a single pool because 
# the rule is "fixed pure combinations of all possible A and X sites"
A_ALL = apply_dummies({"K", "Rb", "Cs", "NH4"}) | apply_dummies({"Na"})

B_LISTS = {
    "A3B2X9": {
        "HIGH":  {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Ga", "Y", "Zr", "Nb", "Mo", "In", "Sb", "Bi", "Al", "As", "Ru", "Rh", "Au", "Eu", "Tb", "Yb", "Nd", "La"},
        "RESTR": {"Si", "Ca", "Sr", "Ba", "Ge", "Sn", "Pb", "Cu", "Zn", "Ag"},
        "X_ALL": apply_dummies({"Cl", "Br", "I", "F", "CN", "SCN"}) | apply_dummies({"O", "S", "Se", "Te"})
    },
    "ABX3": {
        "HIGH":  {"Ca", "Sr", "Ba", "Pb", "Sn", "Ge", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Eu"},
        "RESTR": {"Si", "Sc", "Ti", "V", "Cr", "Y", "Zr", "Nb", "Mo", "Ag", "In", "Sb", "Bi", "Al", "As", "Ru", "Rh", "Au", "Tb", "Yb", "Nd", "La"},
        "X_ALL": apply_dummies({"Cl", "Br", "I", "F", "CN", "SCN"}) | apply_dummies({"O", "S", "Se", "Te"})
    },
    "A2B1B2X6": {
        "HIGH":  {"Ag", "Cu", "Au", "Bi", "Sb", "In", "Ga", "Al", "Sc", "Y", "La", "Nd", "Eu", "Tb", "Yb", "Fe", "Ru", "Rh", "Cr", "V"},
        "RESTR": {"Si", "Ca", "Sr", "Ba", "Pb", "Sn", "Ge", "Ti", "Zr", "Mn", "Co", "Ni", "Zn", "Nb", "Mo", "As"},
        "X_ALL": apply_dummies({"Cl", "Br", "I", "F", "CN", "SCN"}) | apply_dummies({"O", "S", "Se", "Te"})
    },
    "A2BX6": {
        "HIGH":  {"Sn", "Pb", "Ge", "Ti", "Zr", "Ru", "Mn", "Mo", "Se", "Te"},
        "RESTR": {"Si", "Ca", "Sr", "Ba", "Sc", "V", "Cr", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Nb", "Ag", "In", "Sb", "Bi", "Al", "As", "Rh", "Au", "Eu", "Tb", "Yb", "Nd", "La"},
        "X_ALL": apply_dummies({"Cl", "Br", "I", "F", "CN", "SCN"}) | apply_dummies({"O", "S"})
    }
}

# --- 4. GENERATOR HELPER FUNCTIONS ---
def build_pure_dict(element_set):
    """
    Creates a dict of {Element: [0.0, 1.0]}. 
    ProtoCSP will evaluate all one-hot (100% pure) combinations of these.
    """
    return {el: [0.0, 1.0] for el in element_set}

def generate_b_site_pairs(high_prob_set, restricted_set):
    """
    Generates the strict binary configurations for the B-site.
    Returns a list of tuples: (composition_dict, filename_label)
    """
    pairs = []
    high = sorted(list(high_prob_set))
    restr = sorted(list(restricted_set))
    
    # 1. Pure High-Probability (1.0) -> Evaluated as 0.5 + 0.5 of itself
    for h in high:
        pairs.append(({h: [1.0]}, f"Pure_{h}"))
        
    # 2. High/High Binary Alloy (0.5 / 0.5)
    for h1, h2 in combinations(high, 2):
        pairs.append(({h1: [0.5], h2: [0.5]}, f"{h1}50_{h2}50"))
        
    # 3. High/Restricted Binary Alloy (0.75 High / 0.25 Restr)
    # Rule: Never combine two restricted. Restricted is always 0.25.
    for h in high:
        for r in restr:
            pairs.append(({h: [0.75], r: [0.25]}, f"{h}75_{r}25"))
            
    return pairs

# --- 5. MAIN EXECUTION LOOP ---
print("="*60)
print("PROTOCSP WORKFLOW GENERATOR: STRICT PAIRWISE SCREENING")
print("="*60)

output_base_dir = "screening_jsons"
os.makedirs(output_base_dir, exist_ok=True)

for proto_id, proto_data in PROTOTYPES.items():
    b_data = B_LISTS[proto_id]
    
    # Create subfolder for this prototype
    proto_dir = os.path.join(output_base_dir, proto_id)
    os.makedirs(proto_dir, exist_ok=True)
    
    # Pre-calculate the exact valid pairs for this prototype's B-site
    b_site_pairs = generate_b_site_pairs(b_data["HIGH"], b_data["RESTR"])
    
    # Universal A and X configs (Fixed Pure)
    a_config = {
        "substitution_fraction": [1.0],
        "substituents": build_pure_dict(A_ALL)
    }
    x_config = {
        "substitution_fraction": [1.0],
        "substituents": build_pure_dict(b_data["X_ALL"])
    }
    
    json_count = 0
    
    for b_comp, b_label in b_site_pairs:
        config = {
            "_description": f"Screening {proto_id} framework. B-site locked to {b_label}.",
            "experiment_name": f"{proto_data['name']}_{b_label}",
            "base_structure": proto_data["cif"],
            "supercell_dimensions": [2, 2, 2],
            "polyatomic_mapping": {k: v["name"] for k, v in POLYATOMIC_MAPPING.items()},
            "site_substitutions": {
                proto_data["sites"]["A"]: a_config,
                proto_data["sites"]["X"]: x_config
            },
            "samples_per_composition": 1 # SQS is deterministic enough here
        }
        
        # Determine B-site assignment logic
        if proto_id == "A2B1B2X6":
            # DOUBLE PEROVSKITE: B1 is the strict pair, B2 sweeps through pure High elements
            b1_elem, b2_elem = proto_data["sites"]["B"]
            
            config["site_substitutions"][b1_elem] = {
                "substitution_fraction": [1.0],
                "substituents": b_comp
            }
            # B2 sweeps all Pure High elements
            config["site_substitutions"][b2_elem] = {
                "substitution_fraction": [1.0],
                "substituents": build_pure_dict(b_data["HIGH"])
            }
        else:
            # STANDARD PROTOTYPES: Apply the pair config directly to the single B site
            b_site_elem = proto_data["sites"]["B"][0]
            config["site_substitutions"][b_site_elem] = {
                "substitution_fraction": [1.0],
                "substituents": b_comp
            }
            
        # Write the JSON
        filename = f"config_{b_label}.json"
        with open(os.path.join(proto_dir, filename), "w") as f:
            json.dump(config, f, indent=2)
            
        json_count += 1
        
    # Calculate internal structures per JSON for the summary
    a_opts = len(A_ALL)
    x_opts = len(b_data["X_ALL"])
    structs_per_json = a_opts * x_opts
    
    if proto_id == "A2B1B2X6":
        structs_per_json *= len(b_data["HIGH"]) # B2 pure options
        
    print(f"Prototype: {proto_id} ({proto_data['name']})")
    print(f"  -> Generated {json_count} JSONs in ./{proto_dir}/")
    print(f"  -> Each JSON will internally evaluate {structs_per_json} pure AX frameworks.")
    print(f"  -> Total screening volume: {json_count * structs_per_json:,} specific configurations.\n")

print("Run a specific JSON using: python main.py screening_jsons/ABX3/config_Pure_Pb.json --fast --sqs")