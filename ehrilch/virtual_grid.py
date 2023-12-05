from ehrilch import MoleculeSurface


class VirtualPoint:
    def __init__(self, coords, sphere_point):
        self.coords = coords
        self.sphere_point = sphere_point


class VirtualGrid:
    def __init__(self, center_point):
        self.center_point = center_point
        self.surface: MoleculeSurface = None
        self.edge = None
        self.angle = None
        self.envs = []

    def build(self, angle=0.):
        pass

    def add_env(self):
        pass

    def blosum_score(self, other_grid):
        pass

    def geometry_score(self, other_grid):
        pass

