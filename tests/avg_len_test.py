from ehrlich import load_molecule_surface

pdb_name = '8sib'

molecule_surface = load_molecule_surface(f"../assets/molecule_surface/{pdb_name}.pickle")

print(f"avg shrunk edge len: {molecule_surface.average_shrunk_edge_len}")
print(f"avg sphere edge len: {molecule_surface.average_sphere_edge_len}")
