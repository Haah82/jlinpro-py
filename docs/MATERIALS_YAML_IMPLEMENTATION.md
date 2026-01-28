# Materials YAML Implementation

## Overview
Implemented YAML-based material standards for ACI 318 and Eurocode 2, following the same pattern as the existing TCVN materials YAML configuration.

## Files Created

### 1. materials_aci318.yaml
**Location**: `/data/standards/materials_aci318.yaml`

**Content**:
- **Concrete grades** (8 grades): 3000-15000 psi
  - Properties: fc (cylinder strength), Ec (modulus), fr (rupture modulus), lambda (lightweight factor)
  - SI units: MPa for strength values
  
- **Steel grades** (4 grades): Grade40, Grade60, Grade75, Grade80
  - Properties: fy (yield), fu (ultimate), Es (modulus), epsilon_y (yield strain)
  
- **Design parameters**:
  - φ factors (strength reduction) per ACI 21.2
  - β1 factors for stress block per ACI 22.2.2.4.3
  - Minimum reinforcement ratios
  - Shear design coefficients
  - Development length factors
  - Deflection limits per Table 24.2.2
  - Crack control parameters

### 2. materials_ec2.yaml
**Location**: `/data/standards/materials_ec2.yaml`

**Content**:
- **Concrete grades** (14 grades): C12/15 through C90/105
  - Properties: fck, fck_cube, fcm, fctm, fctk_005, fctk_095, Ecm
  - Strain values: epsilon_c1, epsilon_cu1
  
- **Steel grades** (10 grades): ClassA/B/C, B400A/B/C, B500A/B/C, B400
  - Properties: fyk (characteristic yield), ftk (tensile), Es, epsilon_uk, k-ratio
  
- **Design parameters**:
  - γc, γs partial safety factors
  - Minimum cover for exposure classes (XC1-XS3)
  - Stress block parameters (λ, η)
  - Minimum/maximum reinforcement ratios
  - Shear design coefficients
  - Development length factors
  - Crack width limits per Table 7.1N
  - Deflection limits
  - Durability parameters

## Code Updates

### src/design/aci318.py
**Changes**:
1. Added `import yaml` and `from pathlib import Path`
2. Created `_load_aci_materials()` function to read YAML
3. Refactored `MaterialLoaderACI` class:
   - `get_concrete()`: Loads from YAML, supports both "3000psi" and "4000 psi (27.6 MPa)" formats
   - `get_steel()`: Loads from YAML, supports both "Grade60" and "Grade 60 (414 MPa)" formats
   - `list_concrete_grades()`: Returns available grades
   - `list_steel_grades()`: Returns available grades
   - `get_parameters()`: Returns design parameters from YAML
4. Maintained backward compatibility: Created `CONCRETE_ACI` and `STEEL_ACI` dicts from YAML data

### src/design/eurocode2.py
**Changes**:
1. Added `import yaml` and `from pathlib import Path`
2. Created `_load_ec2_materials()` function to read YAML
3. Refactored `MaterialLoaderEC2` class:
   - `get_concrete()`: Loads from YAML, supports both "C30_37" and "C30/37" formats
   - `get_steel()`: Loads from YAML with epsilon_uk conversion (% to decimal)
   - `list_concrete_grades()`: Returns available grades
   - `list_steel_grades()`: Returns available grades
   - `get_parameters()`: Returns design parameters from YAML
   - Added property methods `CONCRETE_GRADES` and `STEEL_GRADES` for legacy compatibility
4. Removed hardcoded dictionaries (previously 12 concrete + 6 steel grades)

## Validation

### Test Results
```bash
pytest tests/ -k "aci or eurocode" --tb=no -q
# Result: 71 passed, 210 deselected, 12 warnings in 3.62s ✅
```

All existing ACI 318 and Eurocode 2 tests pass without modification, confirming backward compatibility.

### Sample Usage
```python
from src.design.aci318 import MaterialLoaderACI
from src.design.eurocode2 import MaterialLoaderEC2

# ACI 318
conc_aci = MaterialLoaderACI.get_concrete("4000 psi (27.6 MPa)")  # Works!
conc_aci2 = MaterialLoaderACI.get_concrete("4000psi")  # Also works!
steel_aci = MaterialLoaderACI.get_steel("Grade60")
params_aci = MaterialLoaderACI.get_parameters()

# Eurocode 2
conc_ec2 = MaterialLoaderEC2.get_concrete("C30/37")  # Works!
conc_ec2_2 = MaterialLoaderEC2.get_concrete("C30_37")  # Also works!
steel_ec2 = MaterialLoaderEC2.get_steel("B500B")
params_ec2 = MaterialLoaderEC2.get_parameters()
```

## Benefits

1. **Centralized Configuration**: All material properties in one YAML file per standard
2. **Easy Maintenance**: Update properties without touching Python code
3. **Extensibility**: Add new grades by editing YAML, no code changes
4. **Consistency**: Same pattern across TCVN, ACI 318, and Eurocode 2
5. **Validation**: Pydantic models ensure type safety
6. **Backward Compatibility**: Existing code continues to work without changes

## File Structure
```
jlinpro-py/
├── data/
│   └── standards/
│       ├── materials_tcvn.yaml      (existing)
│       ├── materials_aci318.yaml    (NEW)
│       └── materials_ec2.yaml       (NEW)
├── src/
│   └── design/
│       ├── aci318.py                (UPDATED)
│       └── eurocode2.py             (UPDATED)
└── docs/
    └── MATERIALS_YAML_IMPLEMENTATION.md (this file)
```

## Standards References

- **ACI 318-25**: Building Code Requirements for Structural Concrete
- **EN 1992-1-1:2004**: Eurocode 2: Design of concrete structures - Part 1-1
- **TCVN 5574:2018**: Vietnamese Standard for Concrete and Reinforced Concrete Structures

## Migration Notes

No migration required! The implementation is backward compatible:
- Old code using hardcoded dictionaries: ✅ Still works
- Old code calling `MaterialLoaderACI.get_concrete()`: ✅ Still works
- Old test code with "4000 psi (27.6 MPa)" format: ✅ Still works
- Old test code with "C30/37" format: ✅ Still works

The YAML files are loaded once at module import time, so there's no performance penalty.

## Future Enhancements

Potential improvements:
1. Add validation schemas for YAML files
2. Add unit tests specifically for YAML loading
3. Consider caching mechanism for large catalogs
4. Add YAML documentation generator from code comments
5. Support for custom user-defined materials via additional YAML files
