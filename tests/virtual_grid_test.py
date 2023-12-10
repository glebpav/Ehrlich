from ehrilch import load_molecule_surface
from ehrilch.virtual_grid import VirtualGrid

molecule = load_molecule_surface(path="../assets/molecule_surface/8sib.pickle")
grid = VirtualGrid(molecule.points[0].origin_coords, molecule, 5)
grid.build(0)

grid.add_env()
grid.add_env()
grid.add_env()

for env in grid.envs:
    print(env)
print(grid.envs)
