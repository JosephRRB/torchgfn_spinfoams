import os
from typing import Literal
from abc import ABC, abstractmethod

import numpy as np
import torch

ROOT_DIR = os.path.abspath(__file__ + "/../../../")


class BaseSpinFoam(ABC):
    def __init__(
            self,
            spin_j: float,
            n_boundary_intertwiners: int,
            n_vertices: int,
            device_str: Literal["cpu", "cuda"] = "cpu",
    ):
        self.device = torch.device(device_str)
        self.n_boundary_intertwiners = n_boundary_intertwiners
        self.n_vertices = n_vertices
        self.spin_j = float(spin_j)
        self.single_vertex_amplitudes = torch.from_numpy(
            _load_vertex_amplitudes(self.spin_j)
        ).to(self.device)

    def estimate_log_sq_ampl_shift(self, frac=0.99):
        max_abs_ampl = torch.max(torch.abs(self.single_vertex_amplitudes))
        estimate_constant_shift = frac*2*self.n_vertices*torch.log(max_abs_ampl)
        return estimate_constant_shift

    def calculate_log_square_amplitudes(self, boundary_intertwiners):
        sf_ampl = self.get_spinfoam_amplitudes(boundary_intertwiners)
        log_sq_ampl = 2*torch.log(torch.abs(sf_ampl))
        return log_sq_ampl

    @abstractmethod
    def get_spinfoam_amplitudes(self, boundary_intertwiners):
        pass


class SingleVertexSpinFoam(BaseSpinFoam):
    def __init__(
            self,
            spin_j: float,
            device_str: Literal["cpu", "cuda"] = "cpu",
    ):
        super().__init__(
            spin_j=spin_j,
            n_boundary_intertwiners=5,
            n_vertices=1,
            device_str=device_str,
        )

    def get_spinfoam_amplitudes(self, boundary_intertwiners):
        indices = [
            boundary_intertwiners[:, i]
            for i in range(self.n_boundary_intertwiners)
        ]
        return self.single_vertex_amplitudes[indices]


class StarModelSpinFoam(BaseSpinFoam):
    def __init__(
            self,
            spin_j: float,
            device_str: Literal["cpu", "cuda"] = "cpu",
    ):
        super().__init__(
            spin_j=spin_j,
            n_boundary_intertwiners=20,
            n_vertices=6,
            device_str=device_str,
        )

    def _get_amplitude_per_star_edge(self, boundary_intertwiners_per_edge):
        n_ints = boundary_intertwiners_per_edge.shape[1]
        indices = [
            boundary_intertwiners_per_edge[:, i]
            for i in range(n_ints)
        ]
        return self.single_vertex_amplitudes[indices]

    def get_spinfoam_amplitudes(self, boundary_intertwiners):
        vertex_1 = self._get_amplitude_per_star_edge(
            boundary_intertwiners[:, :4]
        )
        vertex_2 = self._get_amplitude_per_star_edge(
            boundary_intertwiners[:, 4:8]
        )
        vertex_3 = self._get_amplitude_per_star_edge(
            boundary_intertwiners[:, 8:12]
        )
        vertex_4 = self._get_amplitude_per_star_edge(
            boundary_intertwiners[:, 12:16]
        )
        vertex_5 = self._get_amplitude_per_star_edge(
            boundary_intertwiners[:, 16:20]
        )
        star_amplitudes = torch.einsum(
            "abcde, ie, id, ic, ib, ia -> i",
            self.single_vertex_amplitudes,
            vertex_1, vertex_2, vertex_3, vertex_4, vertex_5
        )
        return star_amplitudes


def _load_vertex_amplitudes(spin_j):
    vertex = np.load(
        f"{ROOT_DIR}/data/EPRL_vertices/Python/Dl_20/vertex_j_{spin_j}.npz"
    )
    return vertex