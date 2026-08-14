
# Colorado Joint Inversion
# Model Dependency Map

Generated:
2026-08-13 16:58:33


==================================================
1. INPUT DATA LAYER
==================================================

Source datasets:

- Gravity observations
- Magnetic observations
- DEM / elevation information
- Well log information
- Structural interpretation inputs
- Seismic event information


These datasets provide the observations and constraints
used throughout the workflow.



==================================================
2. PREPROCESSING LAYER
==================================================

Input preparation:

Gravity:
    ↓
Gravity grid preparation

Magnetic:
    ↓
Magnetic grid preparation

DEM:
    ↓
Elevation correction

Well logs:
    ↓
Validation preparation

Seismic:
    ↓
Structural interpretation preparation



==================================================
3. JOINT INVERSION CORE
==================================================

Authoritative numerical model:

05_INVERSION_CORE/

colorado_joint_final.npz


Contains:

- Density model
- Susceptibility model
- Cell geometry
- Forward predictions
- Observed data
- Residual calculations



==================================================
4. PHYSICAL PROPERTY MODELS
==================================================


Density Model
-------------

Source:

05_INVERSION_CORE/

01_full_joint_inversion.vti


Output:

01_DENSITY_RESULTS/


Contains:

- Density volume
- Density anomalies
- Density visualization products



Susceptibility Model
--------------------

Source:

05_INVERSION_CORE/

01_full_joint_inversion.vti


Output:

02_SUSCEPTIBILITY_RESULTS/


Contains:

- Susceptibility volume
- High susceptibility anomaly products



==================================================
5. STRUCTURAL GEOLOGY INTERPRETATION
==================================================

Structural products are derived from the inversion
results and geological interpretation workflow.


Location:

03_STRUCTURAL_GEOLOGY_RESULTS/


Includes:

- Fault surfaces
- Candidate fault surfaces
- Candidate structural surfaces
- Ridge surface products
- Interactive structural viewers



==================================================
6. VALIDATION LAYER
==================================================

Validation compares inversion results against
available well information.


Location:

04_VALIDATION/


Includes:

- Well RHOB validation
- Density comparison metrics
- Validation visualizations



==================================================
7. SEISMIC INTEGRATION LAYER
==================================================

Seismic information is maintained as an additional
interpretation layer.


Location:

06_SEISMIC_INTEGRATION/


Purpose:

- Structural context
- Event visualization
- Fault relationship analysis



==================================================
8. FINAL ENTERPRISE PACKAGE
==================================================

All approved outputs are organized into:


01_DENSITY_RESULTS

02_SUSCEPTIBILITY_RESULTS

03_STRUCTURAL_GEOLOGY_RESULTS

04_VALIDATION

05_INVERSION_CORE

06_SEISMIC_INTEGRATION

07_DOCUMENTATION

08_QC_REPORTS



==================================================
MODEL GOVERNANCE RULE
==================================================

Authoritative model files remain unchanged.

Derived products, visualizations, and reports must
reference the approved inversion outputs.

