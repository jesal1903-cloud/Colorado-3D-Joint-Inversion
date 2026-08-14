
# Structural Coordinate Mapping Report

## Colorado Joint Inversion Enterprise Model


## 1. Coordinate Authority

Authoritative numerical model:

05_INVERSION_CORE/colorado_joint_final.npz


Visualization model:

05_INVERSION_CORE/01_full_joint_inversion.vti


The VTI represents the inversion model using local coordinates.


---

## 2. Local VTI Coordinate System


Dimensions:

(149, 102, 11)


Bounds:

X:
0.0 to 88800.0

Y:
0.0 to 60600.0

Z:
0.0 to 6000.0


Spacing:

(600.0, 600.0, 600.0)


VTI Origin:

(0.0, 0.0, 0.0)



---

## 3. Global Model Coordinate System


Authoritative model origin:

x0:

[-1.20733000e+07  4.86658343e+06 -3.94093375e+03]


Resolution:

600.0 meters


Cell Shape:

[148 101  10]



---

## 4. Coordinate Transformation


Global coordinates are calculated as:


GLOBAL = LOCAL + MODEL_ORIGIN


Forward conversion:


X_global = X_local + origin_x

Y_global = Y_local + origin_y

Z_global = Z_local + origin_z



Reverse conversion:


X_local = X_global - origin_x

Y_local = Y_global - origin_y

Z_local = Z_global - origin_z



---

## 5. Structural Surface Interpretation


Fault surfaces:

- fault_surfaces.vtp
- candidate_fault_surfaces.vtp
- candidate_structural_surfaces.vtp


These surfaces are aligned with the global model coordinate system.

No CRS reprojection should be applied.



---

## 6. Ridge Surface Status


File:

ridge_surface_authoritative.vtp


Status:

UNDER REVIEW


Reason:

The ridge extraction workflow used:

grid.origin = (0,0,0)


Additional validation is required.



---

## 7. Issue Tracker


| Issue | Status |
|---|---|
| Fault coordinate mismatch | RESOLVED |
| Local/global coordinate mapping | DOCUMENTED |
| CRS reprojection required | NO |
| Ridge coordinate validation | OPEN |



Generated:

PHASE 4 STEP 9

