from pathlib import Path
import pyvista as pv

ROOT = Path(__file__).resolve().parent.parent

volume = pv.read(ROOT / "final_outputs" / "structural_model.vti")
faults = pv.read(ROOT / "final_outputs" / "fault_surfaces.vtp")

plotter = pv.Plotter()

plotter.add_mesh(
    volume.outline(),
    color="black",
)

plotter.add_mesh(
    faults,
    color="red",
    opacity=0.75,
)

plotter.show_grid()

plotter.show()
