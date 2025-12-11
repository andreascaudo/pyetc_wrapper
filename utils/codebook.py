import yaml
from pathlib import Path

# Load codebook.yaml
codebook_path = Path(__file__).parent.parent / "codebook.yaml"
with open(codebook_path, 'r') as f:
    codebook = yaml.safe_load(f)

# Export enums
Obj_SED = codebook['enums']['Obj_SED']
SED_Name = codebook['enums']['SED_Name']
Obj_Spat_Dis = codebook['enums']['Obj_Spat_Dis']
MAG_FIL = codebook['enums']['MAG_FIL']
MAG_SYS = codebook['enums']['MAG_SYS']
INS = codebook['enums']['INS']
CH = codebook['enums']['CH']
AM = codebook['enums']['AM']
FLI = codebook['enums']['FLI']

# Export default full_obs template
TEMPLATE_FULL_OBS = codebook['full_obs']
