
# Colorado Joint Inversion
# Model Governance Report

Generated:
2026-08-13 16:56:18


# 1. Purpose

This document defines the authoritative model outputs,
data ownership, and relationships between final inversion
products in the Colorado Joint Inversion enterprise package.


# 2. Authoritative Model Outputs


## Density Model Authority

Authoritative file:

05_INVERSION_CORE/
01_full_joint_inversion.vti


Dataset:

Density_Contrast_gcc


Description:

The density contrast volume represents the final
joint inversion density model.


Grid:

Dimensions:
149 x 102 x 11

Spacing:
600 m x 600 m x 600 m


Model extent:

X:
0 - 88800 m

Y:
0 - 60600 m

Z:
0 - 6000 m



# 3. Susceptibility Model Authority


Authoritative file:

05_INVERSION_CORE/
01_full_joint_inversion.vti


Dataset:

Susceptibility_SI


Description:

The susceptibility volume represents the final
magnetic property model produced by the joint inversion.

Grid geometry is identical to the density model.



# 4. Complete Inversion Model Authority


Authoritative file:

05_INVERSION_CORE/
colorado_joint_final.npz


Contains:

- rho_model
- rho_cube
- chi_model
- chi_cube
- cell_centers
- model coordinates
- observed gravity data
- predicted gravity data
- gravity residuals
- observed magnetic data
- predicted magnetic data
- magnetic residuals


This file represents the complete numerical inversion state.



# 5. Structural Geology Relationship


Structural outputs are interpreted products
derived from the inversion results.


Location:

03_STRUCTURAL_GEOLOGY_RESULTS/


Includes:

- fault surfaces
- candidate structures
- ridge surfaces
- structural visualization outputs



# 6. Validation Relationship


Validation products compare model outputs against
available well density information.


Location:

04_VALIDATION/



# 7. Seismic Integration Relationship


Seismic data is maintained as an external interpretation
layer.


Location:

06_SEISMIC_INTEGRATION/


Seismic events are used for structural interpretation
and visualization support.



# 8. Model Governance Rule


The files listed as authoritative outputs should be treated
as the approved final model products.

Derived visualizations and reports should reference these
outputs rather than replacing them.

