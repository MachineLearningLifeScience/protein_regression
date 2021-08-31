import Bio.PDB
from Bio.PDB import PDBParser
from Bio.PDB.Residue import Residue
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.SeqUtils import seq1
from os.path import basename 
import numpy as np
from scipy.spatial.distance import euclidean
from typing import List, Tuple
from tqdm import tqdm
from util.dataset_config import DATASET_PDB_LOOKUP
from jacob_temp_code.helper_functions import IUPAC_SEQ2IDX

class ContactMapper:
    def __init__(self, dataset: str, wt_sequence: np.ndarray, pdb_ID: str=None, tri_dist: bool=True, 
                angstrom_threshold: float=5., check_AA: bool=True, verbose: bool=True):
        self.pdb_file = DATASET_PDB_LOOKUP.get(dataset.upper())
        self.verbose = verbose
        self.wt_sequence = wt_sequence
        self.pdb_ID = basename(self.pdb_file).split(".")[0].upper() if not pdb_ID else pdb_ID
        self.tri_dist = tri_dist
        self.angstrom_threshold = angstrom_threshold
        self.check_AA = check_AA
        pdb_filename = "./data/files/cleaned/{}.pdb".format(self.pdb_file)
        self.structure = PDBParser().get_structure(self.pdb_file.upper(), pdb_filename)
        self.model_obj: Model = self.structure[0]
        self.chains = [chain for chain in self.model_obj]
        # only Chain-A is used
        self.chains = [self.chains[0]]
        self.sequence = []
        self.dim = len(self.chains)
        self.centers: list = []
        self.all_coordinates: list = []
        self.distance_matrices: list = self.calc_distance_matrix(tri_dist=tri_dist)
        self.contact_maps: list = [distance_matrix < self.angstrom_threshold for distance_matrix in self.distance_matrices]
        self.contact_map = self.contact_maps[0]
        self.adjacency: List[Tuple[str, List[int]]] = self.generate_adjacency()

    def generate_adjacency(self) -> np.array:
        """compute adjacency from contact map by retrieving indices"""
        contact_indices = [np.where(np.array(row)==True)[0] for row in self.contact_map]
        neighbors = list(zip(self.sequence, contact_indices))
        # build tuple of adjacent residues (residue, contacts: list)
        return neighbors

    @staticmethod
    def get_CA_coords(res: Residue) -> np.ndarray:
        try:
            coord = res["CA"].coord
        except KeyError as _:
            # no C-alpha present, take mean of 3 residues instead
            coord = np.mean(np.array([a.coord for a in res]))
        return coord

    @staticmethod
    def get_RES_coords(res: Residue) -> np.array:
        coord_vec = np.array([atom.coord for atom in res.get_atoms()])
        return coord_vec

    @staticmethod
    def calc_residue_tri_distance(coord_X: np.ndarray, coord_Y: np.ndarray) -> float:
        # take min atom distance between residues see mgpfusion/code/protein.m:494
        d_vec = np.array([[euclidean(c_X, c_Y) for c_Y in coord_Y] for c_X in coord_X])
        if len(d_vec) == 1:
            # last elem 0
            min_dist = np.min(d_vec)
        else:
            min_dist = np.min(d_vec[np.nonzero(d_vec)])
        return min_dist

    def assert_sequence_against_pdb(self, pdb_res, idx: int, overwrite_pdb=True) -> int:
        pdb_residue = IUPAC_SEQ2IDX.get(seq1(pdb_res))
        if pdb_residue != self.wt_sequence[idx]:
            if self.verbose:
                print("Index: {} - PDB: {} -> SEQUENCE: {}".format(idx, pdb_residue, self.wt_sequence[idx]))
            if overwrite_pdb:
                return self.wt_sequence[idx]
        return pdb_residue

    def calculate_chain_distance(self, chain_X: Chain, chain_Y: Chain, tri_res_calculation=True) -> np.array:
        mat = np.zeros((len(chain_X), len(chain_Y)), float)
        t_coord_X = []
        coord_X = []
        skipped_res = 0
        print("len comparison {} vs. {}".format(len(chain_X), len(self.wt_sequence)))
        for res_idx, res_X in tqdm(enumerate(chain_X)):
            if not Bio.PDB.is_aa(res_X) and self.check_AA:
                skipped_res += 1
                continue
            res = self.assert_sequence_against_pdb(pdb_res=res_X.get_resname(), idx=res_idx)
            self.sequence.append(res)
            for res_Y_pos, res_Y in enumerate(chain_Y):
                if not Bio.PDB.is_aa(res_Y) and self.check_AA:
                    continue
                if tri_res_calculation:
                    t_coord_X = self.get_RES_coords(res_X)
                    t_coord_Y = self.get_RES_coords(res_Y)
                    mat[res_idx, res_Y_pos] = self.calc_residue_tri_distance(t_coord_X, t_coord_Y)
                else: # calc using Cα-centers
                    coord_X = self.get_CA_coords(res_X)
                    coord_Y = self.get_CA_coords(res_Y)
                    mat[res_idx, res_Y_pos] = euclidean(coord_X, coord_Y)
            self.centers.append(coord_X)
            self.all_coordinates.append(t_coord_X)
        # with 1 chain this is quadratic - WARN: does not work for multiple chains
        dim_X = len(chain_X) - skipped_res
        mat_resized = np.zeros((dim_X, dim_X), float)
        mat_resized = mat[:dim_X, :dim_X].copy()
        self.sequence = np.array(self.sequence)
        np.testing.assert_array_equal(self.sequence, self.wt_sequence)
        return mat_resized

    def calc_distance_matrix(self, tri_dist) -> np.array:
        """
        Go through all sequences in all chains and build distance matrix by residue 
        :returns : distance matrix as array
        """
        dist_matrix = []
        for idx, chain_X in enumerate(self.chains):
            # calculate distance for each residue in all the given chains
            for idy, chain_Y in enumerate(self.chains[idx:]):
                dist_matrix.append(self.calculate_chain_distance(chain_X, chain_Y, tri_res_calculation=tri_dist))
        return dist_matrix

    def get_min_distance(self):
        return min(min(mat for mat in self.distance_matrices))

    def get_max_distance(self):
        return max(max(mat for mat in self.distance_matrices))