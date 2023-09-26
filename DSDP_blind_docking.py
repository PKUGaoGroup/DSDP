import os
import random
from openbabel import pybel
import openbabel
import skimage
import numpy as np
import pickle
import copy
from argparse import ArgumentParser, Namespace, FileType
from scipy import ndimage
from math import ceil, sin, cos, sqrt, pi
from itertools import combinations
import time

time_start = time.time_ns()

parser = ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='./test_dataset/DSDP_dataset/', help='Path to the dataset file, please put the pdbqt documents of protein and ligand to one folder')
parser.add_argument('--dataset_name', type=str, default='DSDP_dataset', help='Name of the test dataset')
parser.add_argument('--site_path', type=str, default='./results/DSDP_dataset/site_output/', help='Output path of the site')
parser.add_argument('--exhaustiveness', type=int, default=384, help='Number of sampling threads')
parser.add_argument('--search_depth', type=int, default=40, help='Number of sampling steps')
parser.add_argument('--top_n', type=int, default=1, help='Top N results are exported')
parser.add_argument('--out', type=str, default='./results/DSDP_dataset/docking_results/', help='Output path of DSDP')
parser.add_argument('--log', type=str, default='./results/DSDP_dataset/docking_results/', help='Log path of DSDP')
args = parser.parse_args()

class DrugSiteData:  


    def __init__(self, dataset_path, site_path, cache_file, protein_cavity_same_feature=False, resolution_scale=0.5, max_distance=35, max_translation=3, footprint=None, file_format='mol2', use_cache=True, debug_size=None):
        self.resolution_scale = resolution_scale  
        self.max_distance = max_distance          
        self.max_translation = max_translation
        if use_cache and os.path.exists(cache_file):
             loaded = np.load(cache_file, allow_pickle=True)
             self.cache, self.footprint = loaded['cache'], loaded['footprint']
        else:
            box_size = int(np.ceil(2 * max_distance * resolution_scale + 1))  
            if footprint is not None:  
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
            self.cache = []
            names = os.listdir(dataset_path)
            names.sort()
            for i, name in enumerate(names):
                if debug_size and i==debug_size: break    
                print('load '+str(i)+'/'+str(len(names))+' '+name+' ...')   

                protein = os.path.join(dataset_path, name, name + '_protein.pdbqt')
                protein_coordinate =[]
                with open(protein) as protein_file:
                    protein_lines = protein_file.readlines()
                for line in protein_lines:
                    if len(line) > 77 and ('H' not in line[70:80]) and (('A' in line[0]) or ('H' in line[0])):
                        line=line.strip()
                        atom_coordinate = np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])])
                        protein_coordinate.append(atom_coordinate)
                protein_coords = np.stack(protein_coordinate)

                os.system('./protein_feature_tool/protein_feature_tool -i '+ dataset_path + '/' + name + '/' + name + '_protein.pdbqt -o ' + site_path + '/' + name + '_feature_generate.txt -data ./protein_feature_tool/17_FEATURES_DATA.txt')
                feature = os.path.join(site_path, name + '_feature_generate.txt')
                protein_features_part = np.loadtxt(feature)

                os.system('./surface_tool/surface_tool -i '+ dataset_path + '/' + name + '/' + name + '_protein.pdbqt -o ' + site_path + '/' + name + '_surface.txt')
                surface = os.path.join(site_path,  name + '_surface.txt')
                protein_features_surface = np.loadtxt(surface)
                surface_size = protein_features_surface.shape[0]
                protein_features_surface = protein_features_surface.reshape(surface_size,1)
                protein_features = np.hstack((protein_features_part, protein_features_surface))
                centroid = protein_coords.mean(axis=0)
                protein_coords -= centroid

                self.cache.append((protein_coords, protein_features, centroid, name))

            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            np.savez(cache_file, cache=np.array(self.cache, dtype=object), footprint=self.footprint, allow_pickle=True)  
        self.transform_random = 1

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, index):
        def variate(protein_coords, protein_features, centroid, footprint, vmin=0, vmax=1): 
            def make_grid(coords, features, grid_resolution=1.0/2, max_dist=35.0):  
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
                grid_coords = ((coords + max_dist) / grid_resolution).round().astype(int)  
                in_box = ((grid_coords >= 0) & (grid_coords < box_size)).all(axis=1)  
                grid = np.zeros((1, box_size, box_size, box_size, num_features), dtype=np.float32)
                for (x, y, z), f in zip(grid_coords[in_box], features[in_box]):
                    grid[0, x, y, z] += f
                return grid

            receptor_coords = protein_coords

            receptor_grid = make_grid(receptor_coords, protein_features, max_dist=self.max_distance, grid_resolution=1.0/self.resolution_scale)  

            return receptor_grid     

        protein_coords, protein_features, centroid, name = self.cache[index]

        receptor_grid = variate(protein_coords, protein_features,  centroid,  footprint=self.footprint)
        return (receptor_grid, name, centroid)

    def set_transform_random(self, transform_random):
        self.transform_random = transform_random

    def collate(batch):
        receptors, names, centroids = map(list, zip(*batch))
        return np.vstack(receptors), names, np.vstack(centroids)


    def get_pockets_segmentation(self, density, name, centroid, site_path):  
        os.makedirs(os.path.dirname(site_path), exist_ok=True)
        np.printoptions(precision=4)
        aa = density
        np.save(site_path + '/' + name[0] + '.npy', aa)



import torch
class DrugSiteMind(torch.nn.Module):  
    class ConvNorm(torch.nn.Sequential):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding, active):
            super().__init__()
            self.add_module('conv',torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None))
            self.add_module('norm',torch.nn.BatchNorm3d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None))
            if active: self.add_module('relu',torch.nn.ReLU())

    class SiteConvolution(torch.nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            self.block11 = DrugSiteMind.ConvNorm(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, active=1)  
            self.block12 = DrugSiteMind.ConvNorm(out_channels, out_channels, kernel_size=3, stride=1, padding=1, active=1)      
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
            self.block10 = torch.nn.Upsample(size=(out_dimensions, out_dimensions, out_dimensions), mode='trilinear', align_corners=True)  
            self.block11 = DrugSiteMind.ConvNorm(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, active=1)
            self.block12 = DrugSiteMind.ConvNorm(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, active=1)
            self.block13 = DrugSiteMind.ConvNorm(out_channels, out_channels, kernel_size=1, stride=stride, padding=0, active=0)
            self.block20 = torch.nn.Upsample(size=(out_dimensions, out_dimensions, out_dimensions), mode='trilinear', align_corners=True)  
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

    def __init__(self):  
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

    def forward(self, x):  
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


class DrugMain:
    def site(do_train, model_cache_best='./binding_site_model.pth', site_path = args.site_path, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
        def valid(dataset_name = args.dataset_name, debug_size=None, use_cache_data=1, show_loss_epoch=1):  
            print('drug discovery ...')
            data_valid = DrugSiteData(dataset_path =args.dataset_path, site_path = args.site_path, cache_file = site_path + dataset_name +'.npz', use_cache=use_cache_data, debug_size=debug_size)
   
            load_valid = torch.utils.data.DataLoader(data_valid, batch_size=1, collate_fn=DrugSiteData.collate, shuffle=0, batch_sampler=None, pin_memory=0, num_workers=1, drop_last=True)
            data_valid.set_transform_random(transform_random=0)
            if os.path.exists(model_cache_best):
                network = torch.load(model_cache_best).to(device)  
            else:
                raise Exception('can not find model file', model_cache_best)

            network.eval()

            print('drug discovery >>>')
            save_file = 1

            for i, batch in enumerate(load_valid):
                x,  names, centroids = batch
                x = torch.Tensor(x).to(device).permute(0,4,1,2,3)
                o = network(x)
                p = o.permute(0,2,3,4,1)
                densities = p.detach().cpu().numpy()
                data_valid.get_pockets_segmentation(densities, names, centroids, site_path=site_path)
                os.system('./DSDP_blind_docking/DSDP --ligand '+ args.dataset_path + names[0] + '/' + names[0] + '_ligand.pdbqt --protein '+ args.dataset_path + names[0] + '/' + names[0] + '_protein.pdbqt --site_npy ' +  args.site_path + names[0] + '.npy --out ' + args.out + names[0] + '_out.pdbqt --log ' + args.log + names[0] + '_out.log --top_n ' + str(args.top_n))
                os.system
            print('drug discovery !!!')
        train() if do_train else valid()

if __name__ == '__main__':  
    import signal; signal.signal(signal.SIGINT, lambda self,code: os._exit(0) )
    DrugMain.site(do_train=0)

time_end = time.time_ns()
print('Time:', (time_end-time_start)/(10**6))
