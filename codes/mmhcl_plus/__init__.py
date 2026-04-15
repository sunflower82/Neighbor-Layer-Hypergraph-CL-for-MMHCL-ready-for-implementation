"""MMHCL+ Package --- Revision 5.1
Replaces Barlow Twins (D=8192) with VICReg (D=1024).
"""
from .config import VICREG_PROJ_DIM, MMHCLPlusConfig, load_config

# Backward compatibility alias (deprecated --- will be removed in v6)
BARLOW_PROJ_DIM = VICREG_PROJ_DIM
