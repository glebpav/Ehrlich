from ehrilch import load_molecule_surface
from ehrilch.virtual_grid import VirtualGrid

molecule = load_molecule_surface(path="../assets/molecule_surface/8sib.pickle")

grid1 = VirtualGrid(molecule.points[0].origin_coords, molecule, 5)
grid1.build(0)

grid2 = VirtualGrid(molecule.points[100].origin_coords, molecule, 5)
grid2.build(20)

for i in range(3):
    grid1.add_env()
    grid2.add_env()


print(grid1.blosum_score(grid2))
