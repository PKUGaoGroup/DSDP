import os
import random
from openbabel import pybel
import openbabel
import skimage
import numpy as np
import pickle
import copy

from scipy import ndimage
from math import ceil, sin, cos, sqrt, pi
from itertools import combinations


class DrugSiteData:  

    class Feature():  #done: refered tfbio; todo: refer-to equibind
        def __init__(self, atom_codes=None, atom_labels=None, named_properties=None, save_molecule_codes=True, custom_properties=None, smarts_properties=None, smarts_labels=None):
            self.FEATURE_NAMES = []
            if atom_codes is not None:
                if not isinstance(atom_codes, dict):
                    raise TypeError('Atom codes should be dict, got %s instead' % type(atom_codes))
                codes = set(atom_codes.values())
                for i in range(len(codes)):
                    if i not in codes:
                        raise ValueError('Incorrect atom code %s' % i)
                self.NUM_ATOM_CLASSES = len(codes)
                self.ATOM_CODES = atom_codes
                if atom_labels is not None:
                    if len(atom_labels) != self.NUM_ATOM_CLASSES:
                        raise ValueError('Incorrect number of atom labels: %s instead of %s' % (len(atom_labels), self.NUM_ATOM_CLASSES))
                else:
                    atom_labels = ['atom%s' % i for i in range(self.NUM_ATOM_CLASSES)]
                self.FEATURE_NAMES += atom_labels
            else:
                self.ATOM_CODES = {}
                metals = ([3, 4, 11, 12, 13] + list(range(19, 32)) + list(range(37, 51)) + list(range(55, 84)) + list(range(87, 104)))
                atom_classes = [(5, 'B'), (6, 'C'), (7, 'N'), (8, 'O'), (15, 'P'), ([16, 34], 'S'), ([9, 17, 35, 53], 'halogen'), (metals, 'metal')]
                for code, (atom, name) in enumerate(atom_classes):
                    if type(atom) is list:
                        for a in atom:
                            self.ATOM_CODES[a] = code
                    else:
                        self.ATOM_CODES[atom] = code
                    self.FEATURE_NAMES.append(name)
                self.NUM_ATOM_CLASSES = len(atom_classes)
            if named_properties is not None:
                if not isinstance(named_properties, (list, tuple, np.ndarray)):
                    raise TypeError('named_properties must be a list')
                allowed_props = [prop for prop in dir(pybel.Atom) if not prop.startswith('__')]
                for prop_id, prop in enumerate(named_properties):
                    if prop not in allowed_props:
                        raise ValueError('named_properties must be in pybel.Atom attributes, %s was given at position %s' % (prop_id, prop))
                self.NAMED_PROPS = named_properties
            else:
                self.NAMED_PROPS = ['hyb', 'heavydegree', 'heterodegree', 'partialcharge']
            self.FEATURE_NAMES += self.NAMED_PROPS
            if not isinstance(save_molecule_codes, bool):
                raise TypeError('save_molecule_codes should be bool, got %s instead' % type(save_molecule_codes))
            self.save_molecule_codes = save_molecule_codes
            if save_molecule_codes:            
                self.FEATURE_NAMES.append('molcode')  #remember if an atom belongs to the ligand or to the protein
            self.CALLABLES = []
            if custom_properties is not None:
                for i, func in enumerate(custom_properties):
                    if not callable(func):
                        raise TypeError('custom_properties should be list of callables, got %s instead' % type(func))
                    name = getattr(func, '__name__', '')
                    if name == '':
                        name = 'func%s' % i
                    self.CALLABLES.append(func)
                    self.FEATURE_NAMES.append(name)
            if smarts_properties is None:
                self.SMARTS = ['[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]', '[a]', '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]', '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]', '[r]']
                smarts_labels = ['hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring']
            elif not isinstance(smarts_properties, (list, tuple, np.ndarray)):
                raise TypeError('smarts_properties must be a list')
            else:
                self.SMARTS = smarts_properties
            if smarts_labels is not None:
                if len(smarts_labels) != len(self.SMARTS):
                    raise ValueError('Incorrect number of SMARTS labels: %s instead of %s' % (len(smarts_labels), len(self.SMARTS)))
            else:
                smarts_labels = ['smarts%s' % i for i in range(len(self.SMARTS))]
            self.compile_smarts()
            self.FEATURE_NAMES += smarts_labels

        def compile_smarts(self):
            self.__PATTERNS = []
            for smarts in self.SMARTS:
                self.__PATTERNS.append(pybel.Smarts(smarts))

        def encode_num(self, atomic_num):
            if not isinstance(atomic_num, int):
                raise TypeError('Atomic number must be int, %s was given' % type(atomic_num))
            encoding = np.zeros(self.NUM_ATOM_CLASSES)
            try:
                encoding[self.ATOM_CODES[atomic_num]] = 1.0
            except:
                pass
            return encoding

        def find_smarts(self, molecule):
            if not isinstance(molecule, pybel.Molecule):
                raise TypeError('molecule must be pybel.Molecule object, %s was given' % type(molecule))
            features = np.zeros((len(molecule.atoms), len(self.__PATTERNS)))
            for (pattern_id, pattern) in enumerate(self.__PATTERNS):
                atoms_with_prop = np.array(list(*zip(*pattern.findall(molecule))), dtype=int) - 1
                features[atoms_with_prop, pattern_id] = 1.0
            return features

        def get_features(self, molecule, molcode=None):
            if not isinstance(molecule, pybel.Molecule):
                raise TypeError('molecule must be pybel.Molecule object, %s was given' % type(molecule))
            if molcode is None:
                if self.save_molecule_codes is True:
                    raise ValueError('save_molecule_codes is set to True, you must specify code for the molecule')
            elif not isinstance(molcode, (float, int)):
                raise TypeError('motlype must be float, %s was given' % type(molcode))
            coords = []
            features = []
            heavy_atoms = []
            for i, atom in enumerate(molecule):  #ignore hydrogens and dummy atoms (they have atomicnum set to 0)
                if atom.atomicnum > 1:
                    heavy_atoms.append(i)
                    coords.append(atom.coords)
                    features.append(np.concatenate((self.encode_num(atom.atomicnum), [atom.__getattribute__(prop) for prop in self.NAMED_PROPS], [func(atom) for func in self.CALLABLES])))
            coords = np.array(coords, dtype=np.float32)
            features = np.array(features, dtype=np.float32)
            if self.save_molecule_codes:
                features = np.hstack((features, molcode * np.ones((len(features), 1))))
            features = np.hstack([features, self.find_smarts(molecule)[heavy_atoms]])
            if np.isnan(features).any():
                raise RuntimeError('got NaN when calculating features')
            return coords, features

    class Rotator:
        def _rotation_matrix(axis, theta):  #counterclockwise rotation about a given axis by theta radians"""
            if not isinstance(axis, (np.ndarray, list, tuple)):
                raise TypeError('axis must be an array of floats of shape (3,)')
            try:
                axis = np.asarray(axis, dtype=float)
            except ValueError:
                raise ValueError('axis must be an array of floats of shape (3,)')

            if axis.shape != (3,):
                raise ValueError('axis must be an array of floats of shape (3,)')

            if not isinstance(theta, (float, int)):
                raise TypeError('theta must be a float')

            axis = axis / sqrt(np.dot(axis, axis))
            a = cos(theta / 2.0)
            b, c, d = -axis * sin(theta / 2.0)
            aa, bb, cc, dd = a * a, b * b, c * c, d * d
            bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
            return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                             [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                             [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

        ROTATIONS = [_rotation_matrix([1, 1, 1], 0)]  #create matrices for all possible 90* rotations of a box
        for a1 in range(3):  #about X, Y and Z - 9 rotations
            for t in range(1, 4):
                axis = np.zeros(3)
                axis[a1] = 1
                theta = t * pi / 2.0
                ROTATIONS.append(_rotation_matrix(axis, theta))
        for (a1, a2) in combinations(range(3), 2):  #about each face diagonal - 6 rotations
            axis = np.zeros(3)
            axis[[a1, a2]] = 1.0
            theta = pi
            ROTATIONS.append(_rotation_matrix(axis, theta))
            axis[a2] = -1.0
            ROTATIONS.append(_rotation_matrix(axis, theta))
        for t in [1, 2]:  #about each space diagonal - 8 rotations
            theta = t * 2 * pi / 3
            axis = np.ones(3)
            ROTATIONS.append(_rotation_matrix(axis, theta))
            for a1 in range(3):
                axis = np.ones(3)
                axis[a1] = -1
                ROTATIONS.append(_rotation_matrix(axis, theta))

        def rotate(coords, rotation):
            if not isinstance(coords, (np.ndarray, list, tuple)):
                raise TypeError('coords must be an array of floats of shape (N, 3)')
            try:
                coords = np.asarray(coords, dtype=float)
            except ValueError:
                raise ValueError('coords must be an array of floats of shape (N, 3)')
            shape = coords.shape
            if len(shape) != 2 or shape[1] != 3:
                raise ValueError('coords must be an array of floats of shape (N, 3)')
            if isinstance(rotation, int):
                if rotation >= 0 and rotation < len(DrugSiteData.Rotator.ROTATIONS):
                    return np.dot(coords, DrugSiteData.Rotator.ROTATIONS[rotation])
                else:
                    raise ValueError('Invalid rotation number %s!' % rotation)
            elif isinstance(rotation, np.ndarray) and rotation.shape == (3, 3):
                return np.dot(coords, rotation)
            else:
                raise ValueError('Invalid rotation %s!' % rotation)

    def __init__(self, tidy_path, cache_file, site_file='ligand.mol2',site_path='./', protein_cavity_same_feature=False, resolution_scale=0.5, max_distance=35, max_translation=3, footprint=None, file_format='mol2', use_cache=True, debug_size=None):
        self.resolution_scale = resolution_scale  #
        self.max_distance = max_distance          #
        self.max_translation = max_translation
        if use_cache and os.path.exists(cache_file):
             loaded = np.load(cache_file, allow_pickle=True)
             self.cache, self.footprint = loaded['cache'], loaded['footprint']
        else:
            def get_features_binary_cavity(mol, dim=1):  #for cavity6.mol2  #dim should be 18 for Unet
                coords = []
                for atom in mol.atoms:
                    coords.append(atom.coords)
                coords = np.array(coords)
                features = np.ones((len(coords), dim))
                return coords, features

            box_size = int(np.ceil(2 * max_distance * resolution_scale + 1))  #TODO +2?
            if footprint is not None:  #footprint: margin for pocket based on ligand structure
                if isinstance(footprint, int):
                    if footprint == 0:
                        footprint = np.ones([1] * 5)
                    elif footprint < 0:
                        raise ValueError('footprint cannot be negative')
                    elif (2 * footprint + 3) > box_size:
                        raise ValueError('footprint cannot be bigger than box')
                    else:
                        footprint = skimage.draw.ellipsoid(footprint, footprint, footprint)
                        footprint = footprint.reshape((1, *footprint.shape, 1))
                elif isinstance(footprint, np.ndarray):
                    if not ((footprint.ndim == 5) and (len(footprint) == 1) and (footprint.shape[-1] == 1)):
                        raise ValueError('footprint shape should be (1, N, M, L, 1), got %s instead' % str(footprint.shape))
                else:
                    raise TypeError('footprint should be either int or np.ndarray of shape (1, N, M, L, 1), got %s instead' % type(footprint))
                self.footprint = footprint
            else:
                footprint = skimage.draw.ellipsoid(2, 2, 2)
                self.footprint = footprint.reshape((1, *footprint.shape, 1))
            feature = DrugSiteData.Feature(save_molecule_codes=False)  #module 'pybel' has no attribute 'Smarts'  ->  pip uninstall pybel
            self.cache = []
            names = os.listdir(tidy_path)
            names.sort()
            for i, name in enumerate(names):
                if debug_size and i==debug_size: break    
                print('load '+str(i)+'/'+str(len(names))+' '+name+' ...')       
                # try:
                protein = os.path.join(tidy_path, name, name + '_protein.pdb')
                protein_coordinate =[]
                with open(protein) as protein_file:
                    protein_lines = protein_file.readlines()
                for line in protein_lines:
                    if len(line) > 70 and ('H' not in line[70:80]):
                        line=line.strip()
                        atom_coordinate = np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])])
                        protein_coordinate.append(atom_coordinate)
                protein_coords = np.stack(protein_coordinate)

                os.system('../protein_feature_tool/protein_feature_tool -i '+ tidy_path + '/' + name + '/' + name + '_protein.pdb -o ' + tidy_path + '/' + name + '/'+ name + '_feature_generate.txt -data ../protein_feature_tool/17_FEATURES_DATA.txt')

                feature = os.path.join(tidy_path, name, name + '_feature_generate.txt')
                protein_features_part = np.loadtxt(feature)

                os.system('../surface_tool/surface_tool -i '+ tidy_path + '/' + name + '/' + name + '_protein.pdb -o ' + tidy_path + '/' + name + '/'+ name + '_surface.txt')

                surface = os.path.join(tidy_path, name, name + '_surface.txt')
                protein_features_surface = np.loadtxt(surface)
                surface_size = protein_features_surface.shape[0]
                protein_features_surface = protein_features_surface.reshape(surface_size,1)
                protein_features = np.hstack((protein_features_part, protein_features_surface))
                cavity = next(pybel.readfile(file_format, os.path.join(tidy_path, name, site_file)))
                cavity_coords, cavity_features = feature.get_features(cavity) if protein_cavity_same_feature else get_features_binary_cavity(cavity)
                centroid = protein_coords.mean(axis=0)
                protein_coords -= centroid
                cavity_coords -= centroid                      
                self.cache.append((protein_coords, protein_features, cavity_coords, cavity_features, centroid, name))
                # except:
                #     continue
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            np.savez(cache_file, cache=np.array(self.cache, dtype=object), footprint=self.footprint, allow_pickle=True)  #np.savez(<FILE>, <KEY>=<VALUE>, ...)->npz

        self.y_channels = self.cache[0][3].shape[1]  #cavity_features  #36
        self.transform_random = 1

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, index):
        def variate(protein_coords, protein_features, cavity_coords, cavity_features, centroid, rotation, translation, footprint, vmin=0, vmax=1):
            def make_grid(coords, features, grid_resolution=1.0/2, max_dist=35.0):  #output: [G,G,G,F]grid_size = 2 * max_dist / grid_resolution + 1
                try:
                    coords = np.asarray(coords, dtype=float)
                except ValueError:
                    raise ValueError('coords must be an array of floats of shape (N, 3)')
                c_shape = coords.shape
                if len(c_shape) != 2 or c_shape[1] != 3:
                    raise ValueError('coords must be an array of floats of shape (N, 3)')
                N = len(coords)
                try:
                    features = np.asarray(features, dtype=float)
                except ValueError:
                    raise ValueError('features must be an array of floats of shape (N, F)')
                f_shape = features.shape
                if len(f_shape) != 2 or f_shape[0] != N:
                    raise ValueError('features must be an array of floats of shape (N, F)')
                if not isinstance(grid_resolution, (float, int)):
                    raise TypeError('grid_resolution must be float')
                if grid_resolution <= 0:
                    raise ValueError('grid_resolution must be positive')
                if not isinstance(max_dist, (float, int)):
                    raise TypeError('max_dist must be float')
                if max_dist <= 0:
                    raise ValueError('max_dist must be positive')
                num_features = f_shape[1]
                max_dist = float(max_dist)
                grid_resolution = float(grid_resolution)
                box_size = ceil(2 * max_dist / grid_resolution + 1)
                grid_coords = ((coords + max_dist) / grid_resolution).round().astype(int)  #move all atoms to the neares grid point
                in_box = ((grid_coords >= 0) & (grid_coords < box_size)).all(axis=1)  #remove atoms outside the box
                grid = np.zeros((1, box_size, box_size, box_size, num_features), dtype=np.float32)
                for (x, y, z), f in zip(grid_coords[in_box], features[in_box]):
                    grid[0, x, y, z] += f
                return grid

            receptor_coords = DrugSiteData.Rotator.rotate(protein_coords[:], rotation) + translation
            receptor_grid = make_grid(receptor_coords, protein_features, max_dist=self.max_distance, grid_resolution=1.0/self.resolution_scale)  #
     
            pocket_coords = DrugSiteData.Rotator.rotate(cavity_coords[:], rotation) 
            pocket_dens = make_grid(pocket_coords, cavity_features, max_dist=self.max_distance, )  #grid_resolution=1.0/self.resolution_scale  #(-1, 36, 36, 36, 1)  #Convert atom coordinates and features represented as 2D arrays into a fixed-sized 3D box

            margin = ndimage.maximum_filter(pocket_dens, footprint=footprint)
            pocket_dens += margin
            pocket_dens = pocket_dens.clip(vmin, vmax)
            zoom = receptor_grid.shape[1] / pocket_dens.shape[1]
            pocket_dens = np.stack([ndimage.zoom(pocket_dens[0, ..., i], zoom) for i in range(self.y_channels)], -1)
            pocket_dens = np.expand_dims(pocket_dens, 0)
            return receptor_grid, pocket_dens

        protein_coords, protein_features, cavity_coords, cavity_features, centroid, name = self.cache[index]
        rotation =  random.choice(range(0, 24)) if self.transform_random else 0
        translation = self.max_translation * np.random.rand(1, 3) if self.transform_random else (0, 0, 0)
        receptor_grid, cavity_grid = variate(protein_coords, protein_features, cavity_coords, cavity_features, centroid, rotation=rotation, translation=translation, footprint=self.footprint)
        return (receptor_grid, cavity_grid, name, centroid)

    def set_transform_random(self, transform_random):
        self.transform_random = transform_random

    def collate(batch):
        receptors, cavities, names, centroids = map(list, zip(*batch))
        return np.vstack(receptors), np.vstack(cavities), names, np.vstack(centroids)


import torch
class DrugSiteMind(torch.nn.Module):  #https://github.com/jivankandel/PUResNet  #https://gitlab.com/cheminfIBB/kalasanty
    class ConvNorm(torch.nn.Sequential):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding, active):
            super().__init__()
            self.add_module('conv',torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None))
            self.add_module('norm',torch.nn.BatchNorm3d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None))
            if active: self.add_module('relu',torch.nn.ReLU())

    class SiteConvolution(torch.nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            self.block11 = DrugSiteMind.ConvNorm(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, active=1)  #padding=0/valid
            self.block12 = DrugSiteMind.ConvNorm(out_channels, out_channels, kernel_size=3, stride=1, padding=1, active=1)      #padding=1/same
            self.block13 = DrugSiteMind.ConvNorm(out_channels, out_channels, kernel_size=1, stride=1, padding=0, active=0)
            self.block21 = DrugSiteMind.ConvNorm(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, active=0)

        def forward(self, x):
              h11 = self.block11(x)
              h12 = self.block12(h11)
              h13 = self.block13(h12)
              h21 = self.block21(x)
              o = torch.add(h13,h21)       
              o = torch.nn.functional.relu(o)
              return o

    class SiteUpConvolution(torch.nn.Module):
        def __init__(self, in_channels, out_channels, out_dimensions, stride=1):
            super().__init__()
            self.block10 = torch.nn.Upsample(size=(out_dimensions, out_dimensions, out_dimensions), mode='trilinear', align_corners=True)  #size=  | scale_factor=(2, 2, 2)
            self.block11 = DrugSiteMind.ConvNorm(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, active=1)
            self.block12 = DrugSiteMind.ConvNorm(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, active=1)
            self.block13 = DrugSiteMind.ConvNorm(out_channels, out_channels, kernel_size=1, stride=stride, padding=0, active=0)
            self.block20 = torch.nn.Upsample(size=(out_dimensions, out_dimensions, out_dimensions), mode='trilinear', align_corners=True)  #size=  | scale_factor=(2, 2, 2)
            self.block21 = DrugSiteMind.ConvNorm(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, active=0)

        def forward(self, x):
              h10 = self.block10(x)
              h11 = self.block11(h10)
              h12 = self.block12(h11)
              h13 = self.block13(h12)
              h20 = self.block20(x)
              h21 = self.block21(h20)
              o = torch.add(h13,h21)
              o = torch.nn.functional.relu(o)
              return o

    class SiteIdentity(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.block11 = DrugSiteMind.ConvNorm(in_channels, out_channels, kernel_size=1, stride=1, padding=0, active=1)
            self.block12 = DrugSiteMind.ConvNorm(in_channels, out_channels, kernel_size=3, stride=1, padding=1, active=1)
            self.block21 = DrugSiteMind.ConvNorm(in_channels, out_channels, kernel_size=1, stride=1, padding=0, active=0)

        def forward(self, x):
              h11 = self.block11(x)
              h12 = self.block12(h11)
              h21 = self.block21(x)
              o = torch.add(h12,h21)
              o = torch.nn.functional.relu(o)
              return o

    def __init__(self):  #torch.cat  torch.nn.init.  Sequential  ReLU/LeakyReLU/ELU BatchNorm3d/InstanceNorm3d MaxPool3d Upsample/ConvTranspose3d Dropout Softmax
        super().__init__()
        ci=18; c1=ci*1; c2=ci*2; c3=ci*4; c4=ci*8; c5=ci*16

        self.down_c1 = self.__class__.SiteConvolution(18, c1, stride=1)
        self.iden_d1 = self.__class__.SiteIdentity(c1, c1)
        self.iden_x1 = self.__class__.SiteIdentity(c1, c1)

        self.down_c2 = self.__class__.SiteConvolution(c1, c2, stride=2)
        self.iden_d2 = self.__class__.SiteIdentity(c2, c2)
        self.iden_x2 = self.__class__.SiteIdentity(c2, c2)

        self.down_c3 = self.__class__.SiteConvolution(c2, c3, stride=2)
        self.iden_d3 = self.__class__.SiteIdentity(c3, c3)
        self.iden_x3 = self.__class__.SiteIdentity(c3, c3)

        self.down_c4 = self.__class__.SiteConvolution(c3, c4, stride=3)
        self.iden_d4 = self.__class__.SiteIdentity(c4, c4)
        self.iden_x4 = self.__class__.SiteIdentity(c4, c4)

        self.down_c5 = self.__class__.SiteConvolution(c4, c5, stride=3)
        self.iden_d5 = self.__class__.SiteIdentity(c5, c5)  

        #UNET, down(batch:-1, channel:288(c5), dimenson:1,dimenson:1,dimenson:1) to up...

        self.up_c4 = self.__class__.SiteUpConvolution(c5, c5, out_dimensions=3)
        self.iden_u4 = self.__class__.SiteIdentity(c5, c5)

        self.up_c3 = self.__class__.SiteUpConvolution(c5+c4, c4, out_dimensions=9)
        self.iden_u3 = self.__class__.SiteIdentity(c4, c4)

        self.up_c2 = self.__class__.SiteUpConvolution(c4+c3, c3, out_dimensions=18)
        self.iden_u2 = self.__class__.SiteIdentity(c3, c3)

        self.up_c1 = self.__class__.SiteUpConvolution(c3+c2, c2, out_dimensions=36)
        self.iden_u1 = self.__class__.SiteIdentity(c2, c2)
    
        self.out_conv = torch.nn.Conv3d(in_channels=c2+c1, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.out_sigmoid = torch.nn.Sigmoid()

    def forward(self, x):  #permute(0,4,1,2,3):(-1,36,36,36,18)(N,D,H,W,C) -> (-1,18,36,36,36) (N,C,D,H,W)
        hd11 = self.down_c1(x)
        hd12 = self.iden_d1(hd11)
        i1 = self.iden_x1(hd12)

        hd21 = self.down_c2(hd12)
        hd22 = self.iden_d2(hd21)
        i2 = self.iden_x2(hd22)

        hd31 = self.down_c3(hd22)
        hd32 = self.iden_d3(hd31)
        i3= self.iden_x3(hd32)

        hd41 = self.down_c4(hd32)
        hd42 = self.iden_d4(hd41)
        i4 = self.iden_x4(hd42)

        hd51 = self.down_c5(hd42)
        hd52 = self.iden_d5(hd51)

        #UNET, down to up ...

        hu41 = self.up_c4(hd52)
        hu42 = self.iden_u4(hu41)
        hu43 = torch.concat([hu42, i4], dim=1)

        hu31 = self.up_c3(hu43)
        hu32 = self.iden_u3(hu31)
        hu33 = torch.concat([hu32, i3], dim=1)

        hu21 = self.up_c2(hu33)
        hu22 = self.iden_u2(hu21)
        hu23 = torch.concat([hu22, i2], dim=1)

        hu11 = self.up_c1(hu23)
        hu12 = self.iden_u1(hu11)
        hu13 = torch.concat([hu12, i1], dim=1)

        oc = self.out_conv(hu13)
        o = self.out_sigmoid(oc)
        return o  

class DrugSiteLoss(torch.nn.Module):  
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predict, target, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):  #2*(x*y) / ((x+y)+1)  #https://zhuanlan.zhihu.com/p/86704421
        dice = 0.0
        position = torch.arange(0, 36).to(device)
        site_numbers = target.sum(dim=(2,3,4)) +1 
        site_numbers_o = predict.sum(dim=(2,3,4)) +1 
        center_target_x = ((target.sum(dim=(3,4)) * position).sum(2).reshape(target.size(0), target.size(1), 1) / (36 *  site_numbers))
        center_target_y = ((target.sum(dim=(2,4)) * position).sum(2).reshape(target.size(0), target.size(1), 1) / (36 *  site_numbers))
        center_target_z = ((target.sum(dim=(2,3)) * position).sum(2).reshape(target.size(0), target.size(1), 1) / (36 *  site_numbers))

        max_predict_x = predict.max(dim=4).values.max(dim=3).values.argmax(dim=2) / 36.0
        max_predict_y = predict.max(dim=4).values.max(dim=2).values.argmax(dim=2) / 36.0
        max_predict_z = predict.max(dim=3).values.max(dim=2).values.argmax(dim=2) / 36.0
        
        center_p_t = (max_predict_x-center_target_x).pow(2) + (max_predict_y-center_target_y).pow(2) + (max_predict_z-center_target_z).pow(2)
                
        for i in range(predict.size(1)):  #predict = predict.squeeze(dim=1)
            dice += 2 * (predict[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1) / (predict[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + target[:,i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + self.smooth)
        dice = dice / predict.size(1)
        return torch.clamp((1 - dice).mean(), 0, 1) +  0.1 * center_p_t.mean()

class DrugMain:
    def site(do_train, model_cache_best='./ai_drug_site_model_best.pth', site_path='./', device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        def train(dataset='train_dataset', batch_size=2, debug_size=None, epoch0=0, epochs=500, show_loss_epoch=10, reuse_model=0, model_cache_best_prefix='./ai_drug_site_model_best'):
            print('drug discovery ...')
            data_train = DrugSiteData(tidy_path='./'+dataset, cache_file='./ai_drug_site_'+dataset+'.npz', site_file='ligand.mol2',site_path='./', use_cache=1, debug_size=debug_size)
            load_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, collate_fn=DrugSiteData.collate, shuffle=1, batch_sampler=None, pin_memory=0, num_workers=8, drop_last=True)  #DrugSiteData.collate
            data_train.set_transform_random(transform_random=1)
            print('drug discovery >>>')
            if reuse_model and os.path.exists(model_cache_best):
                network = torch.load(model_cache_best).to(device)
            else:
                network = DrugSiteMind().to(device)
            network.train()
            optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
            losser = DrugSiteLoss().to(device)
            loss_best_epoch = -1
            loss_best_train = None
            loss_best_model = None
            for epoch in range(epoch0, epochs):
                losses = []
                for i, batch in enumerate(load_train):
                    x, y, names, centroids = batch
                    x = torch.Tensor(x).to(device).permute(0,4,1,2,3)
                    y = torch.Tensor(y).to(device).permute(0,4,1,2,3)
                    o = network(x)
                    optimizer.zero_grad()
                    loss = losser(o, y)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                    if epoch%show_loss_epoch==0: print('DrugSite: train epoch {:09d}/{:09d}, batch {:04d}/{:04d}, loss {:.4f} ...'.format(epoch, epochs, i, len(load_train), loss.item()))
                loss_this_train = sum(losses)/len(losses)
                if loss_best_train == None or loss_this_train < loss_best_train:
                    if epoch - loss_best_epoch >= 1:
                        loss_best_epoch = epoch
                        loss_best_train = loss_this_train
                        loss_best_model = model_cache_best_prefix+'_'+'{:09d}'.format(epoch)+'_'+'{:2f}'.format(loss_this_train)+'.pth'
                        os.makedirs(os.path.dirname(loss_best_model), exist_ok=True)
                        torch.save(network, loss_best_model)  #torch.save(network.state_dict(),<FILE>)
                        import shutil; shutil.copyfile(loss_best_model, model_cache_best);
                print('train: epoch={:09d}  loss_this_train={:.4f} ***'.format(loss_best_epoch, loss_this_train))
                print()            
            print('train: epoch={:09d}  loss_train_best_end={:.4f} $$$'.format(loss_best_epoch, loss_best_train), '   (copy best model from ' + loss_best_model + ' to ' + model_cache_best + ')','\n')
            print('drug discovery !!!')
            print()

        train() if do_train else valid()

if __name__ == '__main__':  
    import signal; signal.signal(signal.SIGINT, lambda self,code: os._exit(0) )
    DrugMain.site(do_train=1)
