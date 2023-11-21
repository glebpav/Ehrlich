import numpy as np
import torch

ATOM_VWR = {'C': 1.9080, 'H': 1.387, 'N': 1.8240, 'S': 2.0000, 'P': 2.1000, 'O': 1.6612, 'N1+': 1.8240}
ATOM_COLOR = {
    'C': '#A3F06D',
    'H': '#BAD2F0',
    'N': '#5354F0',
    'S': '#F0DE51',
    'P': '#E75AF0',
    'O': '#F04C38',
    'N1+': '#5354F0'
}


class Optimizer:
    def __init__(
            self,
            atoms,
            labels,
            V,
            adj,
            gpu=False,
            history=None,
    ):

        if history is None:
            self.history = {0: [], 1: [], 2: [], 3: []}
        else:
            self.history = history

        self.device = 'cuda:0' if gpu else 'cpu'

        self.atoms = torch.tensor(atoms).to(self.device)
        self.labels = labels
        self.vwrs = torch.tensor([ATOM_VWR[l] for l in labels], dtype=torch.float32)

        self.V = torch.tensor(V).to(self.device)
        self.adj = torch.tensor(self.parse_adj(adj)).to(self.device)

        self.avg_force_array = []

        self.target_edge_len = self.get_edge_len()
        self.VWR = 3 * 2.1

    def parse_adj(self, adj):
        max_adj = max([len(neib) for neib in adj])
        adj_arr = -np.ones((len(adj), max_adj), dtype=np.int64)

        for i in range(len(adj)):
            neibs = adj[i]
            neibs = np.pad(np.array(neibs), (0, max_adj - len(neibs)), constant_values=(0, -1))
            adj_arr[i] = neibs

        return adj_arr

    def optimize(self, **kwargs):
        pgc = torch.mean(self.atoms, dim=0)
        self.atoms -= pgc

        self.shrink(**kwargs)

        self.V += pgc

    def log(self, n, Nsteps):
        r = int(30 * n / Nsteps)
        print(f"\r{n}/{Nsteps} |{'=' * r}>{'.' * (30 - r)}|", end='')

    def get_edge_len(self):
        lens = []
        for i in range(int(0.1 * self.adj.shape[0])):
            a = self.V[i]
            b = self.V[self.adj[i][0]]
            l = np.linalg.norm((a - b).cpu().numpy())
            lens.append(l)

        return np.mean(lens)

    def add_history(self, avg_force, patience, last_weight):

        # [0] - force
        # [1] - filtered_force
        # [2] - differential
        # [3] - avg_differential

        self.history[0].append(avg_force)
        this_step = len(self.history[0])

        if this_step == patience: self.last_avg_force = sum(self.history[0]) / patience
        if this_step >= patience:
            this_weight = 1 - last_weight

            # [1] - filtered_force
            for step_idx in range(len(self.history[1]), this_step):
                self.last_avg_force = self.history[0][step_idx] * this_weight + self.last_avg_force * last_weight
                self.history[1].append(self.last_avg_force)

            # [2] - differential
            for step_idx in range(len(self.history[2]), this_step):
                if step_idx < 1:
                    self.history[2].append(self.history[1][0] - self.history[1][1])
                else:
                    self.history[2].append(self.history[1][step_idx] - self.history[1][step_idx - 1])

            # [3] - avg_differential
            for step_idx in range(len(self.history[3]), this_step):
                self.history[3].append(sum(self.history[2][step_idx - patience: step_idx]) / patience)

    def shrink(self, Nsteps, time_step, lambda_k, close_atoms_ratio, comp_f_ratio, patience, last_weight, accuracy_degree,
               doorstep_accuracy):
        neibs_pad_mask = (self.adj != -1).float()  # v, neibs
        prev_force, prev_edge_grad, _ = self.interaction_gradients(close_atoms_ratio, neibs_pad_mask, comp_f_ratio)

        this_degree = 0
        self.n = 0

        while self.n < Nsteps + 1:
            current_force, current_edge_grad, avg_force = self.interaction_gradients(close_atoms_ratio, neibs_pad_mask,
                                                                                     comp_f_ratio)
            current_force = lambda_k * current_force + (1 - lambda_k) * prev_force
            current_edge_grad = lambda_k * current_edge_grad + (1 - lambda_k) * prev_edge_grad

            # step
            self.V -= time_step * current_force
            self.target_edge_len -= time_step * current_edge_grad

            # rewrite force
            prev_force = current_force
            prev_edge_grad = current_edge_grad

            # collecting statistics
            self.avg_force_array.append(avg_force)
            self.add_history(avg_force, patience, last_weight)

            self.n += 1

            if self.n % 20 == 0:
                # save_data(f"/home/gleb/sasa/data buffer/{pdb_name}/ponts/", f"step{self.n}", self.V)
                pass

            if self.n > patience:
                if abs(self.history[3][-1]) < doorstep_accuracy and this_degree < accuracy_degree:
                    time_step /= 2
                    Nsteps = self.n + 300
                    this_degree += 1
                    doorstep_accuracy /= 5
                    print(f"\naccuracy degree: {this_degree}/{accuracy_degree}, time_step: {format(time_step, '.5g')}")

            self.log(self.n, Nsteps)
        print(f"target edge len: {self.target_edge_len}")

    def interaction_gradients(self, close_atoms_ratio, neibs_pad_mask, comp_f_ratio):
        compression_force, force_avg = self.compression_force(close_atoms_ratio)
        edge_force, target_edge_grad, edge_force_avg = self.edge_force(neibs_pad_mask)

        final_force = comp_f_ratio * compression_force + edge_force
        avg_force = force_avg + edge_force_avg
        return final_force, target_edge_grad, avg_force

    # VERTICES
    def compression_force(self, close_atoms_ratio):
        dist, dif_vecs = self.distance_to_atoms()  # V, atoms
        min_dist = dist.min(dim=-1).values
        close_mask = (dist < close_atoms_ratio * min_dist[..., None]).float()  # v, a view(-1, 1)

        sum_mask = (dist < 100).float()  # v, a view(-1, 1)

        # gradient
        force = (1. - torch.exp((-2.5) * (dist - self.VWR))) / dist  # v, a
        force_avg = torch.mean(force)
        force = dif_vecs * force[..., None]  # v, a, 3

        # mean
        force *= close_mask[..., None]  # v, a, 3
        force = torch.sum(force, dim=1)  # v, 3
        div = torch.sum(close_mask, dim=-1, keepdim=True)  # v, 1
        force = force / div

        return force, force_avg

    def distance_to_atoms(self):
        nv = self.V.shape[0]
        na = self.atoms.shape[0]

        vertixes = self.V.view(-1, 1, 3).repeat((1, na, 1))
        atoms = self.atoms.view(1, -1, 3).repeat((nv, 1, 1))
        dif_vecs = vertixes - atoms

        dist = dif_vecs ** 2
        dist = dist.sum(dim=-1)
        dist = dist.sqrt()

        return dist, dif_vecs

    # EDGES
    def edge_force(self, pad_mask):
        dist, dif_vecs = self.distance_to_neighbours()  # V, neibs
        target_dif = 2 * (dist - self.target_edge_len)
        div = torch.sum(pad_mask, dim=-1, keepdim=True)  # v, 1

        # resize force
        edge_force = target_dif / dist  # v, neibs
        edge_force_avg = torch.sum(edge_force * pad_mask) / torch.sum(div)
        edge_force = edge_force[..., None] * dif_vecs  # v, neibs, 3

        # mean
        edge_force *= pad_mask[..., None]
        edge_force = edge_force.sum(dim=1)  # v, 3
        edge_force = edge_force / div

        # target edge grad
        edge_grad = target_dif * pad_mask
        edge_grad = -edge_grad.sum() / pad_mask.sum()

        return edge_force, edge_grad, edge_force_avg

    def distance_to_neighbours(self):
        adjacent_vertixes = self.V[self.adj]
        dif_vecs = self.V.view(-1, 1, 3) - adjacent_vertixes  # v, neibs, 3

        dist = dif_vecs ** 2
        dist = dist.sum(dim=-1)
        dist = dist.sqrt()

        return dist, dif_vecs
