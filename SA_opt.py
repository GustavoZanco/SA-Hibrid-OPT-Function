
from subprocess import Popen, PIPE, check_output
from math import radians, sin, cos
from tempfile import NamedTemporaryFile, mkdtemp
import os
import random
from datetime import datetime
from shutil import rmtree
NCORES = 8

import numpy as np


from scipy.spatial import ConvexHull

from par_templates import sgsim_par, kt3d_par

    
def run_kt3d(sondcomp_data, x_col, y_col, z_col, data_col, kt3d_out, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, mindat, maxdat, octant_data, hmax, hmin, vert, azim, dip, plunge, nst, nugget, model_par, par_kt3d = "kt3d0.par"):
    a = open(par_kt3d, "w")
    def_par_kt3d = kt3d_par.format(sondcomp_data = sondcomp_data, kt3d_out = kt3d_out, x = x_col, y = y_col, z = z_col, data_col = data_col, nx = nx, ox = ox, xsize = xsize, ny = ny, oy = oy, ysize = ysize, nz = nz, oz = oz, zsize = zsize, mindat = mindat, maxdat = maxdat, octant_data = octant_data, hmax = hmax, hmin = hmin, vert = vert, azim = azim, dip = dip, plunge = plunge, nst = nst, nugget = nugget, model_par = model_par)
    a.write(def_par_kt3d)
    a.close()
    return check_output(["kt3d.exe", par_kt3d])

    
def run_sgsimxbin(nscore_out, x_col, y_col, z_col, data_col, nscore_trn, Zmin, Zmax, sgsim_out, realizations, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, random, mindat, maxdat, nodes, octant_data, hmax, hmin, vert, azim, dip, plunge, nst, nugget, model_par, par_sgsim = "sgsim0.par"):
    f = open(par_sgsim, "w")
    def_par_sgsim = sgsim_par.format(nscore_out = nscore_out, nscore_trn = nscore_trn, sgsim_out = sgsim_out, x = x_col, y = y_col, z = z_col, data_col = data_col, Zmin = Zmin, Zmax = Zmax, realizations = realizations, nx = nx, ox = ox, xsize = xsize, ny = ny, oy = oy, ysize = ysize, nz = nz, oz = oz, zsize = zsize, hull_file = hull_file, random = random, mindat = mindat, maxdat = maxdat, nodes = nodes, octant_data = octant_data, hmax = hmax, hmin = hmin, vert = vert, azim = azim, dip = dip, plunge = plunge, nst = nst, nugget = nugget, model_par = model_par)
    f.write(def_par_sgsim)
    f.close()
    return Popen(["sgsimxbin.exe",], stdin=PIPE, stdout=PIPE).communicate(par_sgsim)[0]

def async_run_sgsimxbin(nscore_out, x_col, y_col, z_col, data_col, nscore_trn, Zmin, Zmax, sgsim_out, realizations, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, random, mindat, maxdat, nodes, octant_data, hmax, hmin, vert, azim, dip, plunge, nst, nugget, model_par, par_sgsim = "sgsim0.par"):
    f = open(par_sgsim, "w")
    def_par_sgsim = sgsim_par.format(nscore_out = nscore_out, nscore_trn = nscore_trn, sgsim_out = sgsim_out, x = x_col, y = y_col, z = z_col, data_col = data_col, Zmin = Zmin, Zmax = Zmax, realizations = realizations, nx = nx, ox = ox, xsize = xsize, ny = ny, oy = oy, ysize = ysize, nz = nz, oz = oz, zsize = zsize, hull_file = hull_file, random = random, mindat = mindat, maxdat = maxdat, nodes = nodes, octant_data = octant_data, hmax = hmax, hmin = hmin, vert = vert, azim = azim, dip = dip, plunge = plunge, nst = nst, nugget = nugget, model_par = model_par)
    f.write(def_par_sgsim)
    f.close()
    return Popen(["sgsimxbin.exe", par_sgsim])    
    
class BlockData(object):
    def __init__(self, data, x0, y0, z0, x_len, y_len, z_len, nx, ny, nz):
        self.data = data
        self.x0, self.y0, self.z0 = float(x0), float(y0), float(z0)
        self.x_len, self.y_len, self.z_len = float(x_len), float(y_len), float(z_len)
        self.nx, self.ny, self.nz = float(nx), float(ny), float(nz)
    def value(self, x, y, z):
        i = int(round((x - self.x0)/self.x_len))
        j = int(round((y - self.y0)/self.y_len))
        k = int(round((z - self.z0)/self.z_len))
        if i <0 or j <0 or k <0 or i>self.nx or j>self.ny or k>self.nz:
            return -999.
        if i == self.data.shape[0]: i = self.data.shape[0] - 1
        if j == self.data.shape[1]: j = self.data.shape[1] - 1
        if k == self.data.shape[2]: k = self.data.shape[2] - 1
        return self.data[i,j,k]
    def __getitem__(self, index):
        return self.value(*index)
    def drillhole(self, collar, attitude, barrel_length, hole_length, nodata=-999.):
        phi, theta = radians(attitude[0]), radians(attitude[1])
        dz = -barrel_length*sin(theta)
        dh = barrel_length*cos(theta)
        dx = dh*sin(phi)
        dy = dh*cos(phi)
        x0, y0, z0 = collar
        hole_data = []
        for i in range(int(hole_length/barrel_length)):
            x = x0 + i*dx + dx/2
            y = y0 + i*dy + dy/2
            z = z0 + i*dz + dz/2
            if x < self.x0 or y < self.y0 or z < self.z0 \
                           or x > x0+self.data.shape[0]*self.x_len\
                           or y > y0+self.data.shape[1]*self.y_len\
                           or z > z0+self.data.shape[2]*self.z_len:
                continue
            sampled_values = self.value(x,y,z)
            if nodata is not None and np.any(sampled_values == nodata): continue
            hole_data.append((x, y, z, sampled_values.mean()))            
        return hole_data
    def random_collar_generator(self, nholes, x_min, y_min, x_max, y_max, z, attitude, barrel_length, hole_length):
        collar = []
        for hole in range(nholes):
            collar.append((x_min + random.random()*(x_max-x_min), y_min + random.random()*(y_max-y_min), z),)
            drillhole = np.array(self.drillhole(collar[hole], attitude, barrel_length, hole_length))
            while not drillhole.any():
                collar[hole] = (x_min + random.random()*(x_max-x_min), y_min + random.random()*(y_max-y_min), z)
                drillhole = np.array(self.drillhole(collar[hole], attitude, barrel_length, hole_length))
        return collar
    def simulate_additional_dholes(self, old_dh, collar, attitude, barrel_length, hole_length, x_col, y_col, z_col, data_col,\
                           Zmin, Zmax, realizations, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, random, mindat, maxdat,\
                           nodes, octant_data, hmax, hmin, vert, azim, dip, plunge, nst, nugget, model_par, nodata=-999.,\
                           new_dh=None, async=False, workfolder="temp_objf",  **filenames):
        sondcomp_data = os.path.join(workfolder, "sondcomp_data")
        nscore_out = os.path.join(workfolder, "nscore_out")
        nscore_trn = os.path.join(workfolder, "nscore_trn")
        nscore_par_filename = os.path.join(workfolder, "nscore_par_filename")
        sgsimxbin_par_filename = os.path.join(workfolder, "sgsimxbin_par_filename")
        sgsim_out = os.path.join(workfolder, "sgsim_out")
        
        simulated_drillholes = []
        for x, y, z in collar:
            simulated_drillholes.extend(self.drillhole((x, y, z), attitude, barrel_length, hole_length, nodata))
        new_dh = new_dh if new_dh is not None else NamedTemporaryFile(delete=False).name
        drillhole_geoEAS(new_dh, simulated_drillholes)
        cat_geoEAS(sondcomp_data, old_dh, new_dh)
        
        run_nscore(sondcomp_data, data_col, nscore_out, nscore_trn, nscore_par_filename)
        if not async:
            run_sgsimxbin(nscore_out, x_col, y_col, z_col, data_col+1, nscore_trn, Zmin, Zmax, sgsim_out, realizations, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, random, mindat, maxdat, nodes, octant_data, hmax, hmin, vert, azim, dip, plunge, nst, nugget, model_par, sgsimxbin_par_filename)
            new_simulated_data = np.fromfile(sgsim_out, dtype=np.float32).reshape([nx, ny, nz, realizations], order='F')
            return new_simulated_data
        else:
            process = async_run_sgsimxbin(nscore_out, x_col, y_col, z_col, data_col+1, nscore_trn, Zmin, Zmax, sgsim_out, realizations, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, random, mindat, maxdat, nodes, octant_data, hmax, hmin, vert, azim, dip, plunge, nst, nugget, model_par, sgsimxbin_par_filename)
            def process_watch():
                if process.poll() is not None:
                    return np.fromfile(sgsim_out, dtype=np.float32).reshape([nx, ny, nz, realizations], order='F')
                else:
                    return None
            return process_watch

    def simulate_additional_dh(self, old_dh, collar, attitude, barrel_length, hole_length, x_col, y_col, z_col, data_col,\
                           Zmin, Zmax, realizations, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, random, mindat, maxdat,\
                           nodes, octant_data, hmax, hmin, vert, azim, dip, plunge, nst, nugget, model_par, nodata=-999.,\
                           new_dh=None,  **filenames):
        workfolder = "temp_objf"
        sondcomp_data = os.path.join(workfolder, "sondcomp_data")
        nscore_out = os.path.join(workfolder, "nscore_out")
        nscore_trn = os.path.join(workfolder, "nscore_trn")
        nscore_par_filename = os.path.join(workfolder, "nscore_par_filename")
        sgsimxbin_par_filename = os.path.join(workfolder, "sgsimxbin_par_filename")
        sgsim_out = os.path.join(workfolder, "sgsim_out")
        
        simulated_drillholes = []
        simulated_drillholes.extend(self.drillhole(collar, attitude, barrel_length, hole_length, nodata))
        new_dh = new_dh if new_dh is not None else NamedTemporaryFile(delete=False).name
        drillhole_geoEAS(new_dh, simulated_drillholes)
        cat_geoEAS(sondcomp_data, old_dh, new_dh)
        
        run_nscore(sondcomp_data, data_col, nscore_out, nscore_trn, nscore_par_filename)
        run_sgsimxbin(nscore_out, x_col, y_col, z_col, data_col+1, nscore_trn, Zmin, Zmax, sgsim_out, realizations, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, random, mindat, maxdat, nodes, octant_data, hmax, hmin, vert, azim, dip, plunge, nst, nugget, model_par, sgsimxbin_par_filename)
        
        new_simulated_data = np.fromfile(sgsim_out, dtype=np.float32).reshape([nx, ny, nz, realizations], order='F')
        return new_simulated_data
    def kriging_additional_dh(self, old_dh, collar, attitude, barrel_length, hole_length, nodata=-999.,\
                           new_dh=None,  **filenames):
        workfolder = "temp_objf/krig"
        sondcomp_data = os.path.join(workfolder, "sondcomp_data")
        kt3d_par_filename = os.path.join(workfolder, "kt3d_par_filename")
        kt3d_out = os.path.join(workfolder, "kt3d_out")
        krig_drillholes = []
        krig_drillholes.extend(self.drillhole(collar, attitude, barrel_length, hole_length, nodata))
        new_dh = new_dh if new_dh is not None else NamedTemporaryFile(delete=False).name
        drillhole_geoEAS(new_dh, krig_drillholes)
        cat_geoEAS(sondcomp_data, old_dh, new_dh)
        run_kt3d(sondcomp_data, x_col, y_col, z_col, data_col, kt3d_out, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, mindat, maxdat, octant_data, hmax, hmin, vert, azim, dip, plunge, nst, nugget, model_par, par_kt3d_filename)
        new_krig_data = np.loadtxt(kt3d_out, skiprows = 4)#.reshape([20, 28, 30], order='F')
        value, kvariance = new_krig_data.T
        kvariance = kvariance.reshape([nx, ny, nz], order='F')
        hull = np.loadtxt("new_hull02.txt", skiprows = 3).reshape([nx, ny, nz], order='F')
        kvariance_hull = data_hull_select(kvariance, hull)
        value = value.reshape([nx, ny, nz], order='F')
        value_hull = data_hull_select(value, hull)
        return value, kvariance_hull



def drillhole_geoEAS(filename, data):
    np.savetxt(filename, data, fmt='%.5e', delimiter='   ', header="""\
AmostrasComp
4
X
Y
Z
Cu""", comments='')


def cat_geoEAS(filename, old_data, new_data):
    with open(old_data) as old, open(new_data) as new, open(filename, 'w') as out:
        out.write(old_data + " contatenated with " + new_data + "\n")
        old.readline()
        new.readline()
        nvar = old.readline()
        if nvar != new.readline():
            raise ValueError
        out.write(nvar)
        for i in range(int(nvar)):
            var = old.readline()
            if var != new.readline():
                raise ValueError
            out.write(var)
        for line in old:
            out.write(line)
        for line in new:
            out.write(line)

def nearest_neigh(data, shape, origin, dims):
    nearest = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                point = np.array((i,j,k))*dims + origin
                dist = np.linalg.norm(data[:,:3] - point, axis = 1)
                nearest[i,j,k] = data[np.argmin(dist), 3]
    return nearest
    
def voronoi_neighborhood(data, shape, origin, dims, nodata = -999.):
    convex = nearest_neigh(data, shape, origin, dims)
    convex[convex != nodata] = 1
    convex[convex == nodata] = 0
    return convex

def point_in_hull(equations, point, tolerance=0):
    return (equations[:,:3].dot(point) + equations[:,3] <= tolerance).all()

def convex_hull(data, shape, origin, dims):
    points_in_hull = np.zeros(shape, dtype = np.int)
    hull_equations = ConvexHull(data).equations
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                point = np.array((i,j,k))*dims + origin + (np.array(dims))/2
                points_in_hull[i,j,k] = 1 if point_in_hull(hull_equations, point, np.linalg.norm(dims)/2) else 0
    return points_in_hull
            
            
def convex_hull_voronoi_intersection(data, shape, origin, dims, nodata = -999.):
    return np.logical_and(convex_hull(data, shape, origin, dims),voronoi_neighborhood(data, shape, origin, dims, nodata))
    
def data_hull_select(data, hull, nodata = -999.):
    selection = np.zeros_like(data)
    selection[hull == 0] = nodata
    selection[hull != 0] = data[hull != 0]
    return selection

def convex_block_variance(data):
    valid_data = data.copy()
    valid_data[valid_data==-999.] = np.NaN
    return np.nansum(np.nanvar(valid_data, axis=3))
    
def convex_block_mean(data):
    valid_data = data.copy()
    valid_data[valid_data==-999.] = np.NaN
    return np.nanmean(np.nanmean(valid_data, axis=3))
    
def convex_block_CV(data):
    valid_data = data.copy()
    valid_data[valid_data==-999.] = np.NaN
    return np.nanmean(np.nanstd(valid_data, axis=3))/np.nanmean(np.nanmean(valid_data, axis=3))
    
def convex_block_variance_krig(data):
    valid_data = data.copy()
    valid_data[valid_data==-999.] = np.NaN
    return np.nanmean(valid_data)
    

def grid_search(f, block_data, old_dh, step, x_min, y_min, x_max, y_max, z, azmin, azmax, dipmin, dipmax, barrel_length, hole_length, x_col, y_col, z_col, data_col, Zmin, Zmax, realizations, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, seed, mindat, maxdat, nodes, octant_data, hmax, hmin, vert, azim, dip, plunge, nst_sim, nugget_sim, model_par_sim, var, valor = 'mean', **kwargs):
    starttime = datetime.now()
    collar = []
    collar.append((x_min, y_min, z, int(azmin + random.random()*(azmax - azmin)), int(dipmin + random.random()*(dipmax - dipmin))))
    print collar
    best_obj = f(block_data.simulate_additional_dholes(old_dh, collar, attitude, barrel_length, hole_length, x_col, y_col, z_col, data_col,\
                           Zmin, Zmax, realizations, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, random, mindat, maxdat,\
                           nodes, octant_data, hmax, hmin, vert, azim, dip, plunge, nst, nugget, model_par, **kwargs))
    print best_obj
    best_collar = collar
    a = collar[0]
    x,y,z,az,dip = a
    while y is not y_max+1:
        while x is not x_max+1:
            collar = []
            collar.append((x+step, y, z, az, dip))
            print collar
            obj = f(block_data.simulate_additional_dholes(old_dh, collar, attitude, barrel_length, hole_length, x_col, y_col, z_col, data_col,\
                           Zmin, Zmax, realizations, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, random, mindat, maxdat,\
                           nodes, octant_data, hmax, hmin, vert, azim, dip, plunge, nst, nugget, model_par, **kwargs))
            print obj
            if obj<best_obj:
                best_obj = obj
                best_collar = collar
            a = collar[0]
            x,y,z,az,dip = a
        collar = []
        collar.append((x_min, y+step, z, az, dip))
        print collar
        obj = f(block_data.simulate_additional_dholes(old_dh, collar, attitude, barrel_length, hole_length, x_col, y_col, z_col, data_col,\
                           Zmin, Zmax, realizations, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, random, mindat, maxdat,\
                           nodes, octant_data, hmax, hmin, vert, azim, dip, plunge, nst, nugget, model_par, **kwargs))
        print obj
        if obj<best_obj:
            best_obj = obj
            best_collar = collar
        a = collar[0]
        x,y,z,az,dip = a
        while x is not x_max+1:
            collar = []
            collar.append((x+step, y, z, az, dip))
            print collar
            obj = f(block_data.simulate_additional_dholes(old_dh, collar, attitude, barrel_length, hole_length, x_col, y_col, z_col, data_col,\
                           Zmin, Zmax, realizations, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, random, mindat, maxdat,\
                           nodes, octant_data, hmax, hmin, vert, azim, dip, plunge, nst, nugget, model_par, **kwargs))
            print obj
            if obj<best_obj:
                best_obj = obj
                best_collar = collar
            a = collar[0]
            x,y,z,az,dip = a
    finaltime = (datetime.now())
    totaltime = finaltime-starttime
    return best_collar, best_obj, totaltime


   
def simulated_annealing(f, block_data, niter, nholes, old_dh, x_min, y_min, x_max, y_max, z, azmin, azmax, dipmin, dipmax, barrel_length, hole_length, x_col, y_col, z_col, data_col,\
                           Zmin, Zmax, realizations, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, seed, mindat, maxdat,\
                           nodes, octant_data, hmax, hmin, vert, azim, dip, plunge, nst_sim, nugget_sim, model_par_sim, var, valor, fast = False, weight = 0, krigvar = False, nst_krig = 0, nugget_krig = 0, model_par_krig = 0, topo_filename = None, **kwargs):
    
    
    simweight = 1 - weight
    best_xy = []
    if topo_filename == None:
        best_xy = block_data.random_collar_generator(nholes, x_min, y_min, x_max, y_max, z, azmin, azmax, dipmin, dipmax, barrel_length, hole_length, valor)
    if topo_filename is not None:
        best_xy = block_data.random_collar_generator(nholes, x_min, y_min, x_max, y_max, z, azmin, azmax, dipmin, dipmax, barrel_length, hole_length, valor, topo_filename)
    print "initial random xy =", best_xy
    if not krigvar:
        best_obj = f(block_data.simulate_additional_dholes(old_dh, best_xy, barrel_length, hole_length, x_col, y_col, z_col, data_col,\
                           Zmin, Zmax, realizations, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, seed, mindat, maxdat,\
                           nodes, octant_data, hmax, hmin, vert, azim, dip, plunge, nst_sim, nugget_sim, model_par_sim, var, valor, **kwargs))
    if krigvar:
        if weight == 1:
            value, kvar = block_data.kriging_additional_dh(old_dh, best_xy, barrel_length, hole_length, x_col, y_col, z_col,\
                           data_col, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, mindat, maxdat,\
                           octant_data, hmax, hmin, vert, azim, dip, plunge, nst_krig, nugget_krig, model_par_krig, var, valor)
            best_obj = convex_block_variance_krig(kvar)
        else:
            value, kvar = block_data.kriging_additional_dh(old_dh, best_xy, barrel_length, hole_length, x_col, y_col, z_col,\
                           data_col, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, mindat, maxdat,\
                           octant_data, hmax, hmin, vert, azim, dip, plunge, nst_krig, nugget_krig, model_par_krig, var, valor)
            best_obj = simweight*(f(block_data.simulate_additional_dholes(old_dh, best_xy, barrel_length, hole_length, x_col, y_col, z_col, data_col,\
                           Zmin, Zmax, realizations, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, seed, mindat, maxdat,\
                           nodes, octant_data, hmax, hmin, vert, azim, dip, plunge, nst_sim, nugget_sim, model_par_sim, var, valor, **kwargs))) + weight*(convex_block_variance_krig(kvar))
    best_obj_global = best_obj
    best_xy_global = []
    for line in best_xy:
        best_xy_global.append((line),)
    print "initial objective function value simulation =", best_obj
    
    starttime = datetime.now()
    
    if not fast: 
        if not krigvar:
            for i in range(niter):
                iter_startime = datetime.now()
                xy = []
                for line in best_xy:
                    xy.append((line),)
                cooling = 1-((i+1)/float(niter))
                parameter = (random.choice(range(nholes)))
                if azmin == azmax:
                    XorY = (random.choice([0, 1, 4]))
                if dipmin == dipmax:
                    XorY = (random.choice([0, 1, 3]))
                if azmin == azmax and dipmin == dipmax:
                    XorY = (random.choice([0, 1]))
                else:
                    XorY = (random.choice([0, 1, 3, 4]))
                if XorY == 0:
                    xy[parameter] = (x_min + random.random()*(x_max-x_min), best_xy[parameter][1], z, best_xy[parameter][3], best_xy[parameter][4])
                if XorY == 1:
                    xy[parameter] = (best_xy[parameter][0], (y_min + random.random()*(y_max-y_min)), z, best_xy[parameter][3], best_xy[parameter][4])
                if XorY == 3:
                    xy[parameter] = (best_xy[parameter][0], best_xy[parameter][1], z, int(azmin + random.random()*(azmax-azmin)), best_xy[parameter][4])
                if XorY == 4:
                    xy[parameter] = (best_xy[parameter][0], best_xy[parameter][1], z, best_xy[parameter][3], int(dipmin + random.random()*(dipmax-dipmin)))
                if topo_filename is not None:
                    topo_data = np.loadtxt(topo_filename, skiprows=1)
                    numrows=len(topo_data)
                    distlist_topo = []
                    for row in range(numrows):
                        distlist_topo.append(((xy[parameter][0]-topo_data[row][0])**2)+((xy[parameter][1]-topo_data[row][1])**2)**0.5)
                    minindex = np.argmin(distlist_topo)
                    distlist_topo[minindex] = 1000000
                    minindex2 = np.argmin(distlist_topo)
                    distlist_topo[minindex2] = 1000000
                    meanz = (topo_data[minindex][2]+topo_data[minindex2][2])/2
                    xy[parameter] = (xy[parameter][0], xy[parameter][1], meanz, xy[parameter][3], xy[parameter][4])
                new_drillhole = np.array(block_data.drillhole(xy[parameter], barrel_length, hole_length, valor))
                while not new_drillhole.any():
                    if XorY == 0:
                        xy[parameter] = (x_min + random.random()*(x_max-x_min), best_xy[parameter][1], z, best_xy[parameter][3], best_xy[parameter][4])
                    if XorY == 1:
                        xy[parameter] = (best_xy[parameter][0], (y_min + random.random()*(y_max-y_min)), z, best_xy[parameter][3], best_xy[parameter][4])
                    if XorY == 3:
                        xy[parameter] = (best_xy[parameter][0], best_xy[parameter][1], z, int(azmin + random.random()*(azmax-azmin)), best_xy[parameter][4])
                    if XorY == 4:
                        xy[parameter] = (best_xy[parameter][0], best_xy[parameter][1], z, best_xy[parameter][3], int(dipmin + random.random()*(dipmax-dipmin)))
                    if topo_filename is not None:
                        topo_data = np.loadtxt(topo_filename, skiprows=1)
                        numrows=len(topo_data)
                        distlist_topo = []
                        for row in range(numrows):
                            distlist_topo.append(((xy[parameter][0]-topo_data[row][0])**2)+((xy[parameter][1]-topo_data[row][1])**2)**0.5)
                        minindex = np.argmin(distlist_topo)
                        distlist_topo[minindex] = 1000000
                        minindex2 = np.argmin(distlist_topo)
                        distlist_topo[minindex2] = 1000000
                        meanz = (topo_data[minindex][2]+topo_data[minindex2][2])/2
                        xy[parameter] = (xy[parameter][0], xy[parameter][1], meanz, xy[parameter][3], xy[parameter][4])
                    new_drillhole = np.array(block_data.drillhole(xy[parameter], barrel_length, hole_length, valor))
                obj = f(block_data.simulate_additional_dholes(old_dh, xy, barrel_length, hole_length, x_col, y_col, z_col, data_col,\
                           Zmin, Zmax, realizations, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, seed, mindat, maxdat,\
                           nodes, octant_data, hmax, hmin, vert, azim, dip, plunge, nst_sim, nugget_sim, model_par_sim, var, valor, **kwargs))
                print "%i) hole=%.1f, index=%.1f, parameter new value =%.5f, obj_sim=%.5f  --- current best: obj_sim=%.5f" % (i, parameter, XorY, xy[parameter][XorY], obj, best_obj)
                if obj < best_obj_global:
                    best_obj_global = obj
                    best_xy_global = xy
                    print "new best global found =", best_obj_global
                    print "#"*100
                if obj < best_obj:
                    best_obj = obj
                    best_xy = xy
                    print "="*100
                if obj > best_obj:
                    control = (random.random())
                    if cooling > control:
                        best_obj = obj
                        best_xy = xy
                        print "*"*100
                print "Iteration %i completed in %s." % (i, datetime.now()-iter_startime)
        if krigvar:
            if weight == 1:
                for i in range(niter):
                    iter_startime = datetime.now()
                    xy = []
                    for line in best_xy:
                        xy.append((line),)
                    cooling = 1-((i+1)/float(niter))
                    parameter = (random.choice(range(nholes)))
                    if azmin == azmax:
                        XorY = (random.choice([0, 1, 4]))
                    if dipmin == dipmax:
                        XorY = (random.choice([0, 1, 3]))
                    if azmin == azmax and dipmin == dipmax:
                        XorY = (random.choice([0, 1]))
                    else:
                        XorY = (random.choice([0, 1, 3, 4]))
                    if XorY == 0:
                        xy[parameter] = (x_min + random.random()*(x_max-x_min), best_xy[parameter][1], z, best_xy[parameter][3], best_xy[parameter][4])
                    if XorY == 1:
                        xy[parameter] = (best_xy[parameter][0], (y_min + random.random()*(y_max-y_min)), z, best_xy[parameter][3], best_xy[parameter][4])
                    if XorY == 3:
                        xy[parameter] = (best_xy[parameter][0], best_xy[parameter][1], z, int(azmin + random.random()*(azmax-azmin)), best_xy[parameter][4])
                    if XorY == 4:
                        xy[parameter] = (best_xy[parameter][0], best_xy[parameter][1], z, best_xy[parameter][3], int(dipmin + random.random()*(dipmax-dipmin)))
                    if topo_filename is not None:
                        topo_data = np.loadtxt(topo_filename, skiprows=1)
                        numrows=len(topo_data)
                        distlist_topo = []
                        for row in range(numrows):
                            distlist_topo.append(((xy[parameter][0]-topo_data[row][0])**2)+((xy[parameter][1]-topo_data[row][1])**2)**0.5)
                        minindex = np.argmin(distlist_topo)
                        distlist_topo[minindex] = 1000000
                        minindex2 = np.argmin(distlist_topo)
                        distlist_topo[minindex2] = 1000000
                        meanz = (topo_data[minindex][2]+topo_data[minindex2][2])/2
                        xy[parameter] = (xy[parameter][0], xy[parameter][1], meanz, xy[parameter][3], xy[parameter][4])
                    new_drillhole = np.array(block_data.drillhole(xy[parameter], barrel_length, hole_length, valor))
                    while not new_drillhole.any():
                        if XorY == 0:
                            xy[parameter] = (x_min + random.random()*(x_max-x_min), best_xy[parameter][1], z, best_xy[parameter][3], best_xy[parameter][4])
                        if XorY == 1:
                            xy[parameter] = (best_xy[parameter][0], (y_min + random.random()*(y_max-y_min)), z, best_xy[parameter][3], best_xy[parameter][4])
                        if XorY == 3:
                            xy[parameter] = (best_xy[parameter][0], best_xy[parameter][1], z, int(azmin + random.random()*(azmax-azmin)), best_xy[parameter][4])
                        if XorY == 4:
                            xy[parameter] = (best_xy[parameter][0], best_xy[parameter][1], z, best_xy[parameter][3], int(dipmin + random.random()*(dipmax-dipmin)))
                        if topo_filename is not None:
                            topo_data = np.loadtxt(topo_filename, skiprows=1)
                            numrows=len(topo_data)
                            distlist_topo = []
                            for row in range(numrows):
                                distlist_topo.append(((xy[parameter][0]-topo_data[row][0])**2)+((xy[parameter][1]-topo_data[row][1])**2)**0.5)
                            minindex = np.argmin(distlist_topo)
                            distlist_topo[minindex] = 1000000
                            minindex2 = np.argmin(distlist_topo)
                            distlist_topo[minindex2] = 1000000
                            meanz = (topo_data[minindex][2]+topo_data[minindex2][2])/2
                            xy[parameter] = (xy[parameter][0], xy[parameter][1], meanz, xy[parameter][3], xy[parameter][4])
                        new_drillhole = np.array(block_data.drillhole(xy[parameter], barrel_length, hole_length, valor))
                    value, kvar = block_data.kriging_additional_dh(old_dh, xy, barrel_length, hole_length, x_col, y_col, z_col,\
                           data_col, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, mindat, maxdat,\
                           octant_data, hmax, hmin, vert, azim, dip, plunge, nst_krig, nugget_krig, model_par_krig, var, valor)
                    obj = convex_block_variance_krig(kvar)
                    print "%i) hole=%.1f, index=%.1f, parameter new value =%.5f, obj_sim=%.5f  --- current best: obj_sim=%.5f" % (i, parameter, XorY, xy[parameter][XorY], obj, best_obj)
                    if obj < best_obj_global:
                        best_obj_global = obj
                        best_xy_global = xy
                        print "new best global found =", best_obj_global
                        print "#"*100
                    if obj < best_obj:
                        best_obj = obj
                        best_xy = xy
                        print "="*100
                    if obj > best_obj:
                        control = (random.random())
                        if cooling > control:
                            best_obj = obj
                            best_xy = xy
                            print "*"*100
                    print "Iteration %i completed in %s." % (i, datetime.now()-iter_startime)
            else:
                for i in range(niter):
                    iter_startime = datetime.now()
                    xy = []
                    for line in best_xy:
                        xy.append((line),)
                    cooling = 1-((i+1)/float(niter))
                    parameter = (random.choice(range(nholes)))
                    if azmin == azmax:
                        XorY = (random.choice([0, 1, 4]))
                    if dipmin == dipmax:
                        XorY = (random.choice([0, 1, 3]))
                    if azmin == azmax and dipmin == dipmax:
                        XorY = (random.choice([0, 1]))
                    else:
                        XorY = (random.choice([0, 1, 3, 4]))
                    if XorY == 0:
                        xy[parameter] = (x_min + random.random()*(x_max-x_min), best_xy[parameter][1], z, best_xy[parameter][3], best_xy[parameter][4])
                    if XorY == 1:
                        xy[parameter] = (best_xy[parameter][0], (y_min + random.random()*(y_max-y_min)), z, best_xy[parameter][3], best_xy[parameter][4])
                    if XorY == 3:
                        xy[parameter] = (best_xy[parameter][0], best_xy[parameter][1], z, int(azmin + random.random()*(azmax-azmin)), best_xy[parameter][4])
                    if XorY == 4:
                        xy[parameter] = (best_xy[parameter][0], best_xy[parameter][1], z, best_xy[parameter][3], int(dipmin + random.random()*(dipmax-dipmin)))
                    if topo_filename is not None:
                        topo_data = np.loadtxt(topo_filename, skiprows=1)
                        numrows=len(topo_data)
                        distlist_topo = []
                        for row in range(numrows):
                            distlist_topo.append(((xy[parameter][0]-topo_data[row][0])**2)+((xy[parameter][1]-topo_data[row][1])**2)**0.5)
                        minindex = np.argmin(distlist_topo)
                        distlist_topo[minindex] = 1000000
                        minindex2 = np.argmin(distlist_topo)
                        distlist_topo[minindex2] = 1000000
                        meanz = (topo_data[minindex][2]+topo_data[minindex2][2])/2
                        xy[parameter] = (xy[parameter][0], xy[parameter][1], meanz, xy[parameter][3], xy[parameter][4])
                    new_drillhole = np.array(block_data.drillhole(xy[parameter], barrel_length, hole_length, valor))
                    while not new_drillhole.any():
                        if XorY == 0:
                            xy[parameter] = (x_min + random.random()*(x_max-x_min), best_xy[parameter][1], z, best_xy[parameter][3], best_xy[parameter][4])
                        if XorY == 1:
                            xy[parameter] = (best_xy[parameter][0], (y_min + random.random()*(y_max-y_min)), z, best_xy[parameter][3], best_xy[parameter][4])
                        if XorY == 3:
                            xy[parameter] = (best_xy[parameter][0], best_xy[parameter][1], z, int(azmin + random.random()*(azmax-azmin)), best_xy[parameter][4])
                        if XorY == 4:
                            xy[parameter] = (best_xy[parameter][0], best_xy[parameter][1], z, best_xy[parameter][3], int(dipmin + random.random()*(dipmax-dipmin)))
                        if topo_filename is not None:
                            topo_data = np.loadtxt(topo_filename, skiprows=1)
                            numrows=len(topo_data)
                            distlist_topo = []
                            for row in range(numrows):
                                distlist_topo.append(((xy[parameter][0]-topo_data[row][0])**2)+((xy[parameter][1]-topo_data[row][1])**2)**0.5)
                            minindex = np.argmin(distlist_topo)
                            distlist_topo[minindex] = 1000000
                            minindex2 = np.argmin(distlist_topo)
                            distlist_topo[minindex2] = 1000000
                            meanz = (topo_data[minindex][2]+topo_data[minindex2][2])/2
                            xy[parameter] = (xy[parameter][0], xy[parameter][1], meanz, xy[parameter][3], xy[parameter][4])
                        new_drillhole = np.array(block_data.drillhole(xy[parameter], barrel_length, hole_length, valor))
                    value, kvar = block_data.kriging_additional_dh(old_dh, xy, barrel_length, hole_length, x_col, y_col, z_col,\
                           data_col, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, mindat, maxdat,\
                           octant_data, hmax, hmin, vert, azim, dip, plunge, nst_krig, nugget_krig, model_par_krig, var, valor)
                    obj = simweight*(f(block_data.simulate_additional_dholes(old_dh, xy, barrel_length, hole_length, x_col, y_col, z_col, data_col,\
                           Zmin, Zmax, realizations, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, seed, mindat, maxdat,\
                           nodes, octant_data, hmax, hmin, vert, azim, dip, plunge, nst_sim, nugget_sim, model_par_sim, var, valor, **kwargs))) + weight*(convex_block_variance_krig(kvar))
                    print "%i) hole=%.1f, index=%.1f, parameter new value =%.5f, obj_sim=%.5f  --- current best: obj_sim=%.5f" % (i, parameter, XorY, xy[parameter][XorY], obj, best_obj)
                    if obj < best_obj_global:
                        best_obj_global = obj
                        best_xy_global = xy
                        print "new best global found =", best_obj_global
                        print "#"*100
                    if obj < best_obj:
                        best_obj = obj
                        best_xy = xy
                        print "="*100
                    if obj > best_obj:
                        control = (random.random())
                        if cooling > control:
                            best_obj = obj
                            best_xy = xy
                            print "*"*100
                    print "Iteration %i completed in %s." % (i, datetime.now()-iter_startime)
    
    if fast:
        if not krigvar:
            for i in range(niter):
                iter_startime = datetime.now()
                xy = []
                for line in best_xy:
                    xy.append((line),)
                cooling = ((float(niter)+1)/(i+1))/(float(niter)) 
                parameter = (random.choice(range(nholes)))
                if azmin == azmax:
                    XorY = (random.choice([0, 1, 4]))
                if dipmin == dipmax:
                    XorY = (random.choice([0, 1, 3]))
                if azmin == azmax and dipmin == dipmax:
                    XorY = (random.choice([0, 1]))
                else:
                    XorY = (random.choice([0, 1, 3, 4]))
                if XorY == 0:
                    xy[parameter] = (x_min + random.random()*(x_max-x_min), best_xy[parameter][1], z, best_xy[parameter][3], best_xy[parameter][4])
                if XorY == 1:
                    xy[parameter] = (best_xy[parameter][0], (y_min + random.random()*(y_max-y_min)), z, best_xy[parameter][3], best_xy[parameter][4])
                if XorY == 3:
                    xy[parameter] = (best_xy[parameter][0], best_xy[parameter][1], z, int(azmin + random.random()*(azmax-azmin)), best_xy[parameter][4])
                if XorY == 4:
                    xy[parameter] = (best_xy[parameter][0], best_xy[parameter][1], z, best_xy[parameter][3], int(dipmin + random.random()*(dipmax-dipmin)))
                if topo_filename is not None:
                    topo_data = np.loadtxt(topo_filename, skiprows=1)
                    numrows=len(topo_data)
                    distlist_topo = []
                    for row in range(numrows):
                        distlist_topo.append(((xy[parameter][0]-topo_data[row][0])**2)+((xy[parameter][1]-topo_data[row][1])**2)**0.5)
                    minindex = np.argmin(distlist_topo)
                    distlist_topo[minindex] = 1000000
                    minindex2 = np.argmin(distlist_topo)
                    distlist_topo[minindex2] = 1000000
                    meanz = (topo_data[minindex][2]+topo_data[minindex2][2])/2
                    xy[parameter] = (xy[parameter][0], xy[parameter][1], meanz, xy[parameter][3], xy[parameter][4])
                new_drillhole = np.array(block_data.drillhole(xy[parameter], barrel_length, hole_length, valor))
                while not new_drillhole.any():
                    if XorY == 0:
                        xy[parameter] = (x_min + random.random()*(x_max-x_min), best_xy[parameter][1], z, best_xy[parameter][3], best_xy[parameter][4])
                    if XorY == 1:
                        xy[parameter] = (best_xy[parameter][0], (y_min + random.random()*(y_max-y_min)), z, best_xy[parameter][3], best_xy[parameter][4])
                    if XorY == 3:
                        xy[parameter] = (best_xy[parameter][0], best_xy[parameter][1], z, int(azmin + random.random()*(azmax-azmin)), best_xy[parameter][4])
                    if XorY == 4:
                        xy[parameter] = (best_xy[parameter][0], best_xy[parameter][1], z, best_xy[parameter][3], int(dipmin + random.random()*(dipmax-dipmin)))
                    if topo_filename is not None:
                        topo_data = np.loadtxt(topo_filename, skiprows=1)
                        numrows=len(topo_data)
                        distlist_topo = []
                        for row in range(numrows):
                            distlist_topo.append(((xy[parameter][0]-topo_data[row][0])**2)+((xy[parameter][1]-topo_data[row][1])**2)**0.5)
                        minindex = np.argmin(distlist_topo)
                        distlist_topo[minindex] = 1000000
                        minindex2 = np.argmin(distlist_topo)
                        distlist_topo[minindex2] = 1000000
                        meanz = (topo_data[minindex][2]+topo_data[minindex2][2])/2
                        xy[parameter] = (xy[parameter][0], xy[parameter][1], meanz, xy[parameter][3], xy[parameter][4])
                    new_drillhole = np.array(block_data.drillhole(xy[parameter], barrel_length, hole_length, valor))
                obj = f(block_data.simulate_additional_dholes(old_dh, xy, barrel_length, hole_length, x_col, y_col, z_col, data_col,\
                           Zmin, Zmax, realizations, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, seed, mindat, maxdat,\
                           nodes, octant_data, hmax, hmin, vert, azim, dip, plunge, nst_sim, nugget_sim, model_par_sim, var, valor, **kwargs))
                print "%i) hole=%.1f, index=%.1f, parameter new value =%.5f, obj_sim=%.5f  --- current best: obj_sim=%.5f" % (i, parameter, XorY, xy[parameter][XorY], obj, best_obj)
                if obj < best_obj_global:
                    best_obj_global = obj
                    best_xy_global = xy
                    print "new best global found =", best_obj_global
                    print "#"*100
                if obj < best_obj:
                    best_obj = obj
                    best_xy = xy
                    print "="*100
                if obj > best_obj:
                    control = (random.random())
                    if cooling > control:
                        best_obj = obj
                        best_xy = xy
                        print "*"*100
                print "Iteration %i completed in %s." % (i, datetime.now()-iter_startime)
        if krigvar:
            if weight == 1:
                for i in range(niter):
                    iter_startime = datetime.now()
                    xy = []
                    for line in best_xy:
                        xy.append((line),)
                    cooling = ((float(niter)+1)/(i+1))/(float(niter)) #f_cooling(i, niter)
                    parameter = (random.choice(range(nholes)))
                    if azmin == azmax:
                        XorY = (random.choice([0, 1, 4]))
                    if dipmin == dipmax:
                        XorY = (random.choice([0, 1, 3]))
                    if azmin == azmax and dipmin == dipmax:
                        XorY = (random.choice([0, 1]))
                    else:
                        XorY = (random.choice([0, 1, 3, 4]))
                    if XorY == 0:
                        xy[parameter] = (x_min + random.random()*(x_max-x_min), best_xy[parameter][1], z, best_xy[parameter][3], best_xy[parameter][4])
                    if XorY == 1:
                        xy[parameter] = (best_xy[parameter][0], (y_min + random.random()*(y_max-y_min)), z, best_xy[parameter][3], best_xy[parameter][4])
                    if XorY == 3:
                        xy[parameter] = (best_xy[parameter][0], best_xy[parameter][1], z, int(azmin + random.random()*(azmax-azmin)), best_xy[parameter][4])
                    if XorY == 4:
                        xy[parameter] = (best_xy[parameter][0], best_xy[parameter][1], z, best_xy[parameter][3], int(dipmin + random.random()*(dipmax-dipmin)))
                    if topo_filename is not None:
                        topo_data = np.loadtxt(topo_filename, skiprows=1)
                        numrows=len(topo_data)
                        distlist_topo = []
                        for row in range(numrows):
                            distlist_topo.append(((xy[parameter][0]-topo_data[row][0])**2)+((xy[parameter][1]-topo_data[row][1])**2)**0.5)
                        minindex = np.argmin(distlist_topo)
                        distlist_topo[minindex] = 1000000
                        minindex2 = np.argmin(distlist_topo)
                        distlist_topo[minindex2] = 1000000
                        meanz = (topo_data[minindex][2]+topo_data[minindex2][2])/2
                        xy[parameter] = (xy[parameter][0], xy[parameter][1], meanz, xy[parameter][3], xy[parameter][4])
                    new_drillhole = np.array(block_data.drillhole(xy[parameter], barrel_length, hole_length, valor))
                    while not new_drillhole.any():
                        if XorY == 0:
                            xy[parameter] = (x_min + random.random()*(x_max-x_min), best_xy[parameter][1], z, best_xy[parameter][3], best_xy[parameter][4])
                        if XorY == 1:
                            xy[parameter] = (best_xy[parameter][0], (y_min + random.random()*(y_max-y_min)), z, best_xy[parameter][3], best_xy[parameter][4])
                        if XorY == 3:
                            xy[parameter] = (best_xy[parameter][0], best_xy[parameter][1], z, int(azmin + random.random()*(azmax-azmin)), best_xy[parameter][4])
                        if XorY == 4:
                            xy[parameter] = (best_xy[parameter][0], best_xy[parameter][1], z, best_xy[parameter][3], int(dipmin + random.random()*(dipmax-dipmin)))
                        if topo_filename is not None:
                            topo_data = np.loadtxt(topo_filename, skiprows=1)
                            numrows=len(topo_data)
                            distlist_topo = []
                            for row in range(numrows):
                                distlist_topo.append(((xy[parameter][0]-topo_data[row][0])**2)+((xy[parameter][1]-topo_data[row][1])**2)**0.5)
                            minindex = np.argmin(distlist_topo)
                            distlist_topo[minindex] = 1000000
                            minindex2 = np.argmin(distlist_topo)
                            distlist_topo[minindex2] = 1000000
                            meanz = (topo_data[minindex][2]+topo_data[minindex2][2])/2
                            xy[parameter] = (xy[parameter][0], xy[parameter][1], meanz, xy[parameter][3], xy[parameter][4])
                        new_drillhole = np.array(block_data.drillhole(xy[parameter], barrel_length, hole_length, valor))
                    value, kvar = block_data.kriging_additional_dh(old_dh, xy, barrel_length, hole_length, x_col, y_col, z_col,\
                           data_col, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, mindat, maxdat,\
                           octant_data, hmax, hmin, vert, azim, dip, plunge, nst_krig, nugget_krig, model_par_krig, var, valor)
                    obj = convex_block_variance_krig(kvar)
                    print "%i) hole=%.1f, index=%.1f, parameter new value =%.5f, obj_sim=%.5f  --- current best: obj_sim=%.5f" % (i, parameter, XorY, xy[parameter][XorY], obj, best_obj)
                    if obj < best_obj_global:
                        best_obj_global = obj
                        best_xy_global = xy
                        print "new best global found =", best_obj_global
                        print "#"*100
                    if obj < best_obj:
                        best_obj = obj
                        best_xy = xy
                        print "="*100
                    if obj > best_obj:
                        control = (random.random())
                        if cooling > control:
                            best_obj = obj
                            best_xy = xy
                            print "*"*100
                    print "Iteration %i completed in %s." % (i, datetime.now()-iter_startime)
            else:
                for i in range(niter):
                    iter_startime = datetime.now()
                    xy = []
                    for line in best_xy:
                        xy.append((line),)
                    cooling = ((float(niter)+1)/(i+1))/(float(niter)) #f_cooling(i, niter)
                    parameter = (random.choice(range(nholes)))
                    if azmin == azmax:
                        XorY = (random.choice([0, 1, 4]))
                    if dipmin == dipmax:
                        XorY = (random.choice([0, 1, 3]))
                    if azmin == azmax and dipmin == dipmax:
                        XorY = (random.choice([0, 1]))
                    else:
                        XorY = (random.choice([0, 1, 3, 4]))
                    if XorY == 0:
                        xy[parameter] = (x_min + random.random()*(x_max-x_min), best_xy[parameter][1], z, best_xy[parameter][3], best_xy[parameter][4])
                    if XorY == 1:
                        xy[parameter] = (best_xy[parameter][0], (y_min + random.random()*(y_max-y_min)), z, best_xy[parameter][3], best_xy[parameter][4])
                    if XorY == 3:
                        xy[parameter] = (best_xy[parameter][0], best_xy[parameter][1], z, int(azmin + random.random()*(azmax-azmin)), best_xy[parameter][4])
                    if XorY == 4:
                        xy[parameter] = (best_xy[parameter][0], best_xy[parameter][1], z, best_xy[parameter][3], int(dipmin + random.random()*(dipmax-dipmin)))
                    if topo_filename is not None:
                        topo_data = np.loadtxt(topo_filename, skiprows=1)
                        numrows=len(topo_data)
                        distlist_topo = []
                        for row in range(numrows):
                            distlist_topo.append(((xy[parameter][0]-topo_data[row][0])**2)+((xy[parameter][1]-topo_data[row][1])**2)**0.5)
                        minindex = np.argmin(distlist_topo)
                        distlist_topo[minindex] = 1000000
                        minindex2 = np.argmin(distlist_topo)
                        distlist_topo[minindex2] = 1000000
                        meanz = (topo_data[minindex][2]+topo_data[minindex2][2])/2
                        xy[parameter] = (xy[parameter][0], xy[parameter][1], meanz, xy[parameter][3], xy[parameter][4])
                    new_drillhole = np.array(block_data.drillhole(xy[parameter], barrel_length, hole_length, valor))
                    while not new_drillhole.any():
                        if XorY == 0:
                            xy[parameter] = (x_min + random.random()*(x_max-x_min), best_xy[parameter][1], z, best_xy[parameter][3], best_xy[parameter][4])
                        if XorY == 1:
                            xy[parameter] = (best_xy[parameter][0], (y_min + random.random()*(y_max-y_min)), z, best_xy[parameter][3], best_xy[parameter][4])
                        if XorY == 3:
                            xy[parameter] = (best_xy[parameter][0], best_xy[parameter][1], z, int(azmin + random.random()*(azmax-azmin)), best_xy[parameter][4])
                        if XorY == 4:
                            xy[parameter] = (best_xy[parameter][0], best_xy[parameter][1], z, best_xy[parameter][3], int(dipmin + random.random()*(dipmax-dipmin)))
                        if topo_filename is not None:
                            topo_data = np.loadtxt(topo_filename, skiprows=1)
                            numrows=len(topo_data)
                            distlist_topo = []
                            for row in range(numrows):
                                distlist_topo.append(((xy[parameter][0]-topo_data[row][0])**2)+((xy[parameter][1]-topo_data[row][1])**2)**0.5)
                            minindex = np.argmin(distlist_topo)
                            distlist_topo[minindex] = 1000000
                            minindex2 = np.argmin(distlist_topo)
                            distlist_topo[minindex2] = 1000000
                            meanz = (topo_data[minindex][2]+topo_data[minindex2][2])/2
                            xy[parameter] = (xy[parameter][0], xy[parameter][1], meanz, xy[parameter][3], xy[parameter][4])
                        new_drillhole = np.array(block_data.drillhole(xy[parameter], barrel_length, hole_length, valor))
                    value, kvar = block_data.kriging_additional_dh(old_dh, xy, barrel_length, hole_length, x_col, y_col, z_col,\
                           data_col, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, mindat, maxdat,\
                           octant_data, hmax, hmin, vert, azim, dip, plunge, nst_krig, nugget_krig, model_par_krig, var, valor)
                    obj = simweight*(f(block_data.simulate_additional_dholes(old_dh, xy, barrel_length, hole_length, x_col, y_col, z_col, data_col,\
                           Zmin, Zmax, realizations, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, seed, mindat, maxdat,\
                           nodes, octant_data, hmax, hmin, vert, azim, dip, plunge, nst_sim, nugget_sim, model_par_sim, var, valor, **kwargs))) + weight*(convex_block_variance_krig(kvar))
                    print "%i) hole=%.1f, index=%.1f, parameter new value =%.5f, obj_sim=%.5f  --- current best: obj_sim=%.5f" % (i, parameter, XorY, xy[parameter][XorY], obj, best_obj)
                    if obj < best_obj_global:
                        best_obj_global = obj
                        best_xy_global = xy
                        print "new best global found =", best_obj_global
                        print "#"*100
                    if obj < best_obj:
                        best_obj = obj
                        best_xy = xy
                        print "="*100
                    if obj > best_obj:
                        control = (random.random())
                        if cooling > control:
                            best_obj = obj
                            best_xy = xy
                            print "*"*100
                    print "Iteration %i completed in %s." % (i, datetime.now()-iter_startime)

    if not krigvar:
        best_obj = f(block_data.simulate_additional_dholes(old_dh, best_xy, barrel_length, hole_length, x_col, y_col, z_col, data_col,\
                           Zmin, Zmax, realizations, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, seed, mindat, maxdat,\
                           nodes, octant_data, hmax, hmin, vert, azim, dip, plunge, nst_sim, nugget_sim, model_par_sim, var, valor, **kwargs))
        best_obj_global = f(block_data.simulate_additional_dholes(old_dh, best_xy_global, barrel_length, hole_length, x_col, y_col, z_col, data_col,\
                           Zmin, Zmax, realizations, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, seed, mindat, maxdat,\
                           nodes, octant_data, hmax, hmin, vert, azim, dip, plunge, nst_sim, nugget_sim, model_par_sim, var, valor, **kwargs))
    if krigvar:
        if weight == 1:
            value, kvar = block_data.kriging_additional_dh(old_dh, best_xy, barrel_length, hole_length, x_col, y_col, z_col,\
                           data_col, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, mindat, maxdat,\
                           octant_data, hmax, hmin, vert, azim, dip, plunge, nst_krig, nugget_krig, model_par_krig, var, valor)
            best_obj = convex_block_variance_krig(kvar)
            value, kvar = block_data.kriging_additional_dh(old_dh, best_xy_global, barrel_length, hole_length, x_col, y_col, z_col,\
                           data_col, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, mindat, maxdat,\
                           octant_data, hmax, hmin, vert, azim, dip, plunge, nst_krig, nugget_krig, model_par_krig, var, valor)
            best_obj_global = convex_block_variance_krig(kvar)
        else:
            value, kvar = block_data.kriging_additional_dh(old_dh, best_xy, barrel_length, hole_length, x_col, y_col, z_col,\
                           data_col, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, mindat, maxdat,\
                           octant_data, hmax, hmin, vert, azim, dip, plunge, nst_krig, nugget_krig, model_par_krig, var, valor)
            best_obj = simweight*(f(block_data.simulate_additional_dholes(old_dh, best_xy, barrel_length, hole_length, x_col, y_col, z_col, data_col,\
                           Zmin, Zmax, realizations, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, seed, mindat, maxdat,\
                           nodes, octant_data, hmax, hmin, vert, azim, dip, plunge, nst_sim, nugget_sim, model_par_sim, var, valor, **kwargs))) + weight*(convex_block_variance_krig(kvar))
            value, kvar = block_data.kriging_additional_dh(old_dh, best_xy_global, barrel_length, hole_length, x_col, y_col, z_col,\
                           data_col, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, mindat, maxdat,\
                           octant_data, hmax, hmin, vert, azim, dip, plunge, nst_krig, nugget_krig, model_par_krig, var, valor)
            best_obj_global = simweight*(f(block_data.simulate_additional_dholes(old_dh, best_xy_global, barrel_length, hole_length, x_col, y_col, z_col, data_col,\
                           Zmin, Zmax, realizations, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, seed, mindat, maxdat,\
                           nodes, octant_data, hmax, hmin, vert, azim, dip, plunge, nst_sim, nugget_sim, model_par_sim, var, valor, **kwargs))) + weight*(convex_block_variance_krig(kvar))
    finaltime = (datetime.now())
    totaltime = finaltime-starttime
    print "Total optimization completed in %s." % (datetime.now()-starttime,)
    return best_xy, best_obj, best_xy_global, best_obj_global, str(totaltime)
    
    
    
def auto_mutiple_opt(opt, f, block_data, niter, nholes, old_dh, x_min, y_min, x_max, y_max, z, azmin, azmax, dipmin, dipmax, barrel_length, hole_length, x_col, y_col, z_col, data_col, Zmin, Zmax, realizations, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, seed, mindat, maxdat, nodes, octant_data, hmax, hmin, vert, azim, dip, plunge, nst_sim, nugget_sim, model_par_sim, var, valor = 'mean', **kwargs):
    filename = str(opt).split()[1]+"_"+str(f).split()[1]+"_result.txt"
    for n in nholes:
        best_xy, best_obj, best_xy_global, best_obj_global, time = opt(f, block_data, niter, n, old_dh, x_min, y_min, x_max, y_max, z, azmin, azmax, dipmin, dipmax, barrel_length, hole_length, x_col, y_col, z_col, data_col, Zmin, Zmax, realizations, nx, ox, xsize, ny, oy, ysize, nz, oz, zsize, hull_file, seed, mindat, maxdat, nodes, octant_data, hmax, hmin, vert, azim, dip, plunge, nst_sim, nugget_sim, model_par_sim, var, valor, **kwargs)
        results = "Optimum configuration found for "+str(n)+" holes was "+str(best_xy)+", with "+str(best_obj)+" as objective function result in "+str(niter)+" iterations, that took "+str(time)+ ", using the "+str(opt).split()[1]+" algorithm and "+str(f).split()[1]+" as objective function.\nThe best global configuration found for "+str(n)+" holes was "+str(best_xy_global)+", with "+str(best_obj_global)+" as objective function result in "+str(niter)+" iterations, that took "+str(time)+", using the "+str(opt).split()[1]+" algorithm and "+str(f).split()[1]+" as objective function.\n"
        txt = open(filename, "a+")
        txt.write(results)
        txt.close()
        new_data = []
        for holes in best_xy:
            new_data.extend(block_data.drillhole(holes, barrel_length, hole_length, valor))
        autoname_file = str(opt).split()[1]+"_"+str(n)+"_holes.txt"
        final_filename = str(opt).split()[1]+"_"+str(n)+"_complete.txt"
        drillhole_geoEAS(autoname_file, new_data, var)
        cat_geoEAS(final_filename, old_dh, (autoname_file))
        new_data2 = []
        for data in best_xy_global: 
            new_data2.extend(block_data.drillhole(data, barrel_length, hole_length, valor))
        autoname_file2 = str(opt).split()[1]+"_"+str(n)+"_holes_global.txt"
        final_filename2 = str(opt).split()[1]+"_"+str(n)+"_complete_global.txt"
        drillhole_geoEAS(autoname_file2, new_data2, var)
        cat_geoEAS(final_filename2, old_dh, autoname_file2)