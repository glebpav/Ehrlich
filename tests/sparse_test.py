from ehrilich import read_pdb

molecule = read_pdb("../assets/8sib.pdb")
print(f"molecule coords:\n{molecule.coords}")
print(f"molecule radius: {molecule.get_radius()}")
print(f"count of atoms: {len(molecule.coords)}")

new_molecule = molecule.sparse()
print(f"new molecule coords: {new_molecule.coords}")
print(f"new molecule atoms count: {len(new_molecule.coords)}")

