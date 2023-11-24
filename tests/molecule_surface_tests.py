from ehrilich.molecule import read_pdb
from ehrilich.molecule_surface import make_surface

pdb_name = "8sib.pdb"

molecule = read_pdb(f"../assets/{pdb_name}")

molecule_surface = make_surface(molecule=molecule, n_steps=100)
molecule_surface.save(f"../assets/molecule_surfaces/{pdb_name}")
