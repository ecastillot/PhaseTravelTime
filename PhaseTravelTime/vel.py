import pandas as pd
import pykonal
import os
import math
import numpy as np
from scipy.interpolate import interp1d
from pykonal.transformations import sph2geo,geo2sph
import matplotlib.pyplot as plt
import matplotlib
# import utils as ut
from  . import utils as ut

class Vel3DModel():
    """
    Builds a 3D grid in spherical coordinates with interpolated 
    velocity values from specified  1d velocity model in depth.
    THe grid is builded according to specific latitudes, longitudes, 
    and depth dimensions and  1d velocity model in depth
    """

    def __init__(self,vel1dfile,min_lat,max_lat,min_lon,max_lon,
                min_depth,max_depth, ndepth,nlat,nlon,radial=6371.0):
        """
        Parameters:
        -----------
        vel1dfile: str
            csv path of the 1d velocity model (headers: dep,vp,vs)
        min_lat: float
            Minimum latitude
        max_lat: float
            Maximum latitude
        min_lon: float
            Minimum longitude
        max_lon: float
            Maximum longitude
        min_depth: float
            Minimum depth
        max_depth: float
            Maximum depth
        ndepth: float
            Number of points in depth
        nlat: float
            Number of points in latitude 
        nlon: float
            Number of points in longitude 
        radial: float ; default: 6371.0
            Earth radious
        Attributes:
        ----------
        min_coords: array
            Minimum coordinates in spherical coordinates (rho,theta,phi)
            0 <= rho <= Earth_radious
            0 <= theta <= 360
            0 <= phi <= 180
        node_intervals: array
            Intervals in spherical coordinates (rho,theta,phi)
            rho in km; lat in radians, lon in radians
        """
        self.vel1dfile = vel1dfile
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.nlat = nlat
        self.nlon = nlon
        self.ndepth = ndepth
        self.radial = radial
        self.npts = np.array([ndepth,nlat,nlon])
    
    @property
    def min_coords(self):
        min_coords = geo2sph((self.max_lat,self.min_lon,self.max_depth))
        return min_coords

    @property
    def node_intervals(self):
        node_intervals =  [(self.max_depth-self.min_depth),
                        np.deg2rad(self.max_lat-self.min_lat),
                        np.deg2rad(self.max_lon-self.min_lon)]
        node_intervals = node_intervals/(self.npts-1)

        return node_intervals

    def __prepare_dir2save(self,filepath):
        """
        Parameters:
        filepath: file path will be saved
        Returns:
        Make the directories needed to save the file.
        If the file is already exist, then ask to the user if want to replace it.
        """

        dirpath = os.path.dirname(filepath)
        if os.path.isdir(dirpath ) == False:
            os.makedirs(dirpath)
        else:
            pass

        if os.path.isfile(filepath) == True:
            while True:
                inp = input(f"{filepath} is already created. Dou you want to replace it? (y or n)")
                if inp.upper() == "Y":
                    os.remove(filepath)
                    break
                elif inp.upper() == "N":
                    break
                else:
                    pass

    def get_values(self,phase):
        """
        Parameters
        ----------
        phase: str
            p or s
        Returns:
        --------
        It interpolates the phase velocity values in the builded grid.
        The phase velocity model was previously mentioned in the csv path when initialized the class.
        """
        values = np.zeros(self.npts)
        vel1d = pd.read_csv(self.vel1dfile)
        interp = interp1d(self.radial-vel1d['dep'],
                    vel1d[f'v{phase.lower()}'])
        radii = np.linspace(self.max_depth,self.min_depth,self.npts[0])
        interp_vel = interp(self.radial-radii)
        for ii in range(self.npts[0]):
            values[ii,:,:] = interp_vel[ii]

        return values

    def get_model(self,phase,out=None):
        """
        Parameters
        ----------
        phase: str
            p or s
        out: str
            path to export the velocity model in hdf5 files.
        Returns:
        --------
        Gets the velocity model.
        """
        self.phase = phase
        field = pykonal.fields.ScalarField3D(coord_sys='spherical')
        field.npts = self.npts
        field.min_coords = self.min_coords
        field.node_intervals = self.node_intervals
        field.values = self.get_values(phase)

        if out != None:
            self.__prepare_dir2save(out)
            field.to_hdf(out)
        
        return field

class ReadVel3DModel():
    """
    Util class to understand the velocity model created
    from Vel3D class
    """

    def __init__ (self,models):
        """
        Parameters:
        -----------
        models: dict
            key: str: "Name of the model"
            value: str: HD5F model path 
        Attributes:
        models: dict
            key: str: "Name of the model"
            value: Pykonal Field3D Model: Model loaded from hd5_path
        limits: dict
            key: str: "Name of the model"
            value: 2-tuple: Minimum and Maximum geographic coordinates
        node_intervals: dict
            key: str: "Name of the model"
            value: 3-tuple: Z,lat,lon steps in kilometers 
        
        """
        self._models = models
    
    @property
    def models(self):
        models = {}
        for key,model in self._models.items():
            models[key] = pykonal.fields.read_hdf(model)  
        return models

    @property
    def limits(self):
        limits = {}
        for key,model in self.models.items():
            mincoord = sph2geo(model.min_coords)
            maxcoord = sph2geo(model.max_coords)
            limits[key] = (mincoord,maxcoord)
        return limits

    @property
    def node_intervals(self):
        node_intervals = {}
        for key,model in self.models.items():
            z_step,lat_step,lon_step = model.node_intervals
            node_intervals[key] = [z_step,np.rad2deg(lat_step)*111,
                                    np.rad2deg(lon_step)*111] #*111 in km
        return node_intervals

    def plot_1Dmodel(self,save=None):
        """
        Plot 1D velocity model in depth.
        Parameters:
        -----------
        save: str
            Path to save the figure
        """

        def prepare_graph(key):
            '''Prepare data to plot 1d velocity model'''
            min_coord,max_coord = self.limits[key]
            z_min = min_coord[2]
            z_max = max_coord[2]
            # print(f"{key} 1D model",model)
            z_npts = self.models[key].npts[0]
            depth = np.linspace(z_min ,z_max,z_npts)
            vel1d = self.models[key].values[:,1,1]
            return vel1d,depth


        plt.figure(figsize=(3,6))

        for key in self._models.keys(): 
            vel1d,depth = prepare_graph(key)
            plt.plot(vel1d,depth,label=key)

        plt.gca().invert_yaxis()
        plt.legend()
        plt.ylabel('z [km]')
        plt.xlabel(f'v [km/s]')

        if save != None:
            if os.path.isdir(os.path.dirname(save)) == False:
                os.makedirs(os.path.dirname(save))
            else:   pass

            plt.savefig(save)
        else:
            plt.show()

    def _make_3Dplot(self,ax,ngraph,fig=None ):
        """
        util method that will be used to plot 3D velocity model.
        Parameters:
        -----------
        ax: Matplotlib axis
            3D axis
        ngraph: int
            Number (id) of the graph
        fig: Matplotlib Figure
            Figure
        """

        jet_cmap = matplotlib.cm.jet
        keylist = list(self._models.keys())
        keygraph =keylist[ngraph-1]
        node_intervals = self.node_intervals[keygraph]
        min_coords,max_coords = self.limits[keygraph]
        model = self.models[keygraph]
        x,y,z = model.npts
        values = model.values
        print(f"{keygraph} 3D model",model)
        X,Y,Z = np.mgrid[0:x:1,
                    0:y:1,
                    0:z:1]
        vel = ax.scatter(X, Y, Z, c=values, cmap=jet_cmap.reversed())

        
        base = np.array([[x+(x/4),0,0],[0,y +(y*2),0],[0,0,z+(z/2)]])
        for v in base:
            a = ut.Arrow3D([0,v[0]],[0,v[1]],[0,v[2]],
                    mutation_scale=20,
                    arrowstyle="-|>",lw=3,
                    linestyle='dashed',
                    color="black")
            ax.add_artist(a)

        ax.set_xlabel(f'nz * {round(node_intervals[0],3)} [km]')
        ax.set_ylabel(f'nlat * {round(node_intervals[1],3)} [km]')
        ax.set_zlabel(f'nlon * {round(node_intervals[2],3)} [km]')
        ax.set_title(keygraph)

        if fig != None:
            fig.colorbar(vel, ax=ax,fraction=0.03, pad=0.15)
            fig.suptitle(f'Geo_origin: {min_coords}\n Geo_end: {max_coords}', 
                            fontsize=20)
            fig.suptitle(f'Geo_origin: {min_coords}', fontsize=20)

    def plot_3Dmodel(self,save=None):
        """
        Plot 3D velocity model.
        Parameters:
        -----------
        save: str
            Path to save the figure
        """
        ngraphs = len(self._models.keys())

        ncols = 3
        if ngraphs > ncols:
            nrows = ngraphs/ncols
            nrows = int(math.ceil(nrows))
            fsize = (12,12)
        else:
            nrows = ngraphs
            ncols= 1
            fsize = (7,12)

        fig, axes = plt.subplots(nrows, ncols,subplot_kw={"projection": "3d"},
                                figsize=fsize)

        if len(axes.shape) == 1:
            for ngraph,ax in enumerate(axes,1):
                self._make_3Dplot(ax,ngraph,fig  )
        else:
            for nrow,row_ax in enumerate(axes,1):
                for ncol,col_ax in enumerate(row_ax,1):
                    ngraph = nrow*ncol
                    self._make_3Dplot(col_ax,ngraph,fig )
        
        if save != None:
            if os.path.isdir(os.path.dirname(save)) == False:
                os.makedirs(os.path.dirname(save))
            else:   pass

            plt.savefig(save)
        else:
            plt.show()
        # # fig.suptitle(f'Origin: {self.geo_origin}', fontsize=20)
        return fig

if __name__ == "__main__":

    lats = -5
    latn = 14
    lonw = -80
    lone = -65
    dep0 = -5    
    dep1 = 200
    # ndepth = 61              # number of points in z
    # nlat = 51                # number of points in latitude
    # nlon = 51              # number of points in longitude
    ndepth = 206              # number of points in z
    nlat = 251               # number of points in latitude
    nlon = 251               # number of points in longitude

    vel1dfile = '/home/emmanuel/PEPA/data/vel1dmodel.csv'
    vel3d = Vel3DModel(vel1dfile,lats,latn,lonw,lone,
                    dep0,dep1,ndepth,nlat,nlon)


    P = "/home/emmanuel/results/PEPA/test/vel/P_VelModel.h5"
    S = "/home/emmanuel/results/PEPA/test/vel/S_VelModel.h5"


    # P = "/home/emmanuel/data/P_VelModel.h5"
    p_model = vel3d.get_model("P",P )
    s_model = vel3d.get_model("S",S)

    # read_vel = ReadVel3DModel({"P":P,"S":S})
    # limits = read_vel.limits
    # node_intervals = read_vel.node_intervals
    # read_vel.plot_1Dmodel("figures/vel_zmodel.png")
    # read_vel.plot_3Dmodel("figures/vel_model.png")