from ehrilch import load_molecule_surface

molecule = load_molecule_surface(path="../assets/molecule_surface/8sib.pdb")
molecule.project()
molecule.save("../assets/molecule_surface/8sib.pickle")
