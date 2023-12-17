import ehrilch
from ehrilch import load_molecule_surface
from ehrilch.utils.visualize import Visualize

"""molecule = ehrilch.read_pdb("../assets/4ins2.pdb")
surface = ehrilch.make_surface(molecule)
surface.project()
surface.save("../assets/molecule_surface/4ins2.pickle")"""

# surface1 = load_molecule_surface(path="../assets/molecule_surface/8sib.pickle")
# surface2 = load_molecule_surface(path="../assets/molecule_surface/8sib.pickle")

# visualizer = Visualize(surface1, surface2, surface1.points[0], surface1.points[4], 10000)
# figure = visualizer.draw_align()

surface1 = load_molecule_surface(path="../assets/molecule_surface/4ins2.pickle")
surface2 = load_molecule_surface(path="../assets/molecule_surface/4ins2.pickle")

visualizer = Visualize(surface1, surface2, surface1.points[0], surface1.points[4], 1000)
figure = visualizer.draw_region(opacity=0.3)
