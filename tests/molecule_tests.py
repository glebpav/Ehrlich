from ehrlich.molecule import read_pdb

molecule = read_pdb("../assets/8sib.pdb")
print(f"molecule coords:\n{molecule.coordinates}")
print(f"molecule radius: {molecule.get_radius()}")
print(f"count of atoms: {len(molecule.coordinates)}")

for i in range(200, 210):
    print(f"{i} - name: {molecule.atoms[i].name}; residue: {molecule.atoms[i].residue}")
    print(f"{i} - name: {molecule[i].name}; residue: {molecule[i].residue}")

for atom_idx, atom in enumerate(molecule):
    print(f"atom idx: {atom_idx}; atom name: {atom.name}")
    if atom_idx == 10:
        break
