from ehrilch import load_molecule_surface
from ehrilch.segment import make_sphere_segments

pdb_name = '8sib'

molecule_surface = load_molecule_surface(f"../assets/molecule_surface/{pdb_name}.pickle")
segment_list = make_sphere_segments(molecule_surface, fi=30, k=1.3)

print(len(segment_list))
for segment in segment_list:
    print("center points: ", segment.center_point)
    print("points: ", segment.points)
    print("count of points: ", len(segment.points))
    print("segment_counter: ", segment.amines)
    print("\n")




