import numpy as np


class Sphere:

    def __init__(self, R):
        self.R = R
        self.init_octahedron()
        self.adj = self.get_adjacency()

    def split(self):
        self.split_polygons()
        self.adj = self.get_adjacency()
        return self.V.shape[0], self.get_edge_len()

    def get_adjacency(self):
        adj = [set() for _ in range(self.V.shape[0])]
        for i, j in self.E:
            adj[i].add(j)
            adj[j].add(i)
        return [tuple(s) for s in adj]

    def split_polygons(self):
        e2v = {}
        new_V = [v for v in self.V]
        adj = {}
        new_F = []

        for a, b, c in self.F:
            tmp_idx = []

            for v1, v2 in ((a, b), (b, c), (c, a)):

                if (v1, v2) in e2v:  # this edge already was splitted
                    idx = e2v[(v1, v2)]
                else:
                    new_V.append(self.split_edge((v1, v2)))
                    idx = len(new_V) - 1
                    adj[idx] = set()
                    adj[idx].update({v1, v2})

                if len(tmp_idx) > 0:  # not first edge in this face
                    adj[idx].add(tmp_idx[-1])  # connect to previous
                    adj[tmp_idx[-1]].add(idx)  # connect previous to current
                    new_F.append((tmp_idx[-1], v1, idx))  # add face

                if (v1, v2) not in e2v:  # cache splitted edge
                    e2v[(v1, v2)] = idx
                    e2v[(v2, v1)] = idx

                tmp_idx.append(idx)

            # add last two edges faces and edge
            adj[tmp_idx[-1]].add(tmp_idx[0])
            adj[tmp_idx[0]].add(tmp_idx[-1])

            new_F.append((a, tmp_idx[0], tmp_idx[-1]))  # add side face
            new_F.append(tuple(tmp_idx))  # add middle face

        self.V = np.array(new_V)
        self.F = new_F

        new_E = set()
        for i in adj:
            for j in adj[i]:
                if (j, i) not in new_E:
                    new_E.add((i, j))
        self.E = list(new_E)

    def split_edge(self, e):
        v1, v2 = e
        d1 = self.V[v1]
        d2 = self.V[v2]

        vec = d1 + (d2 - d1) / 2
        vec = vec * (self.R / np.linalg.norm(vec))

        return vec

    def get_edge_len(self):
        idx1, idx2 = self.E[0]
        vec = self.V[idx1] - self.V[idx2]
        return np.linalg.norm(vec)

    def init_octahedron(self):
        V, E, F = self.octahedron(self.R)
        self.V = V
        self.E = E
        self.F = F

    def octahedron(self, R):
        s = R * np.sqrt(2) / 2

        V = np.array([
            [-s, s, 0],
            [s, s, 0],
            [s, -s, 0],
            [-s, -s, 0],
            [0, 0, R],
            [0, 0, -R]
        ], dtype=np.float32)

        E = [(0, 1), (1, 2), (2, 3), (3, 0),
             (0, 4), (1, 4), (2, 4), (3, 4),
             (0, 5), (1, 5), (2, 5), (3, 5)]

        F = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4),
             (0, 1, 5), (1, 2, 5), (2, 3, 5), (3, 0, 5)]

        return V, E, F
