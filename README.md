Usage
===
Read from pdb file
```
import ehrlich as erl
import naskit as nsk


with nsk.pdbRead("prot.pdb") as f:
    pdb = f.read()
    
prot = pdb[0][0]
prot
>>> ProteinChain with 172 AminoacidResidue residues at 0x7f49c3e5e6c0

struct = erl.from_pdb(prot)
```

Make mesh and parse into segments
```
struct.make_mesh(poly_area=25)
struct.project() # finds closest atom for each mesh vertex
struct.make_segments(area=225)
erl.save(struct, "new_struct.pkl")
```

Compare to saved structure
```
# load precomputed structure
ref = erl.load("precomp.pkl")
```

First fast segment comparison
```
def first_filter_func(s1, s2):
    # uses s1.amin_sim(s2) to compare aminoacids
    # uses s.concavity and s.curvature geometry descriptors
    if segments similar enough:
        return True
    return False
    
seg_pairs = []
for i, s1 in enumerate(struct.segments):
    for j, s2 in enumerate(ref.segments):
        if first_filter_func(s1, s2):
            seg_pairs.append((i, j))
```

Second segment comparison
```
def second_filter_func(s1, s2):
    amin_sim, mean_dist = s1.align_compare(s2)
    if segments similar enough:
        return True
    return False
    
similar_pairs = []
for i, j in seg_pairs:
    if second_filter_func(s1.segments[i], s2.segments[j]):
        similar_pairs.append((i, j))
```

Align molecules on similar segments...