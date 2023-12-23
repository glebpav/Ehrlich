from ehrlich import load_molecule_surface
from ehrlich.segment import make_sphere_segments

pdb_name = '8sib'

molecule_surface = load_molecule_surface(f"../assets/molecule_surface/{pdb_name}.pickle")

target_area = 10000

area1 = molecule_surface.sphere_area(target_area)

print(f"target_area: {target_area}")
print(f"area1: {area1}")

segment_list = make_sphere_segments(molecule_surface, area=area1, k=1.3)

print(len(segment_list))
for segment in segment_list:
    print("center points: ", segment.center_point)
    print("points: ", segment.points)
    print("count of points: ", len(segment.points))
    print("segment_counter: ", segment.amines)
    print("\n")


for idx in range(1, 10):
    print(f"segment similarity {0} - {idx}: {segment_list[0].amin_similarity(segment_list[idx])}")




