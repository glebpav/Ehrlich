import ehrlich
from ehrlich import load_molecule_surface
from ehrlich.utils.visualize import Visualize

"""molecule = ehrlich.read_pdb("../assets/4ins2.pdb")
surface = ehrlich.make_surface(molecule)
surface.project()
surface.save("../assets/molecule_surface/4ins2.pickle")"""

"""surface1 = load_molecule_surface(path="../assets/molecule_surface/8sib.pickle")
surface2 = load_molecule_surface(path="../assets/molecule_surface/8sib.pickle")

visualizer = Visualize(surface1, surface2, surface1.points[0], surface1.points[4], 10000)
figure = visualizer.draw_align()"""

surface1 = load_molecule_surface(path="../assets/molecule_surface/8sib.pickle")
surface2 = load_molecule_surface(path="../assets/molecule_surface/8sib.pickle")

visualizer = Visualize(surface1, surface2, surface1.points[7], surface1.points[4], 100000)
figure = visualizer.draw_region(opacity=0.1)
