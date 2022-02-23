# from vel import ReadVel3DModel
# import utils as ut
from . import utils as ut
from .vel import ReadVel3DModel

import ast
import os
import h5py
import math
import numpy as np
import pandas as pd
import pykonal
import concurrent.futures as cf
from pykonal.transformations import sph2geo,geo2sph
import datetime as dt
import time
import warnings
from tables import NaturalNameWarning
warnings.filterwarnings('ignore', category=NaturalNameWarning)
from scipy.spatial import distance
from scipy import interpolate as interp
from scipy.sparse import coo_matrix
import logging
from obspy.geodetics.base import gps2dist_azimuth
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
                   datefmt='%m-%d %H:%M') 

def interp2d(data):
    rows,row_pos = np.unique(data[:,0], return_inverse=True)
    cols,col_pos = np.unique(data[:,1], return_inverse=True)

    pivot_table = np.zeros((len(rows), len(cols)), dtype=data.dtype)
    pivot_table[row_pos, col_pos] = data[:, 2]

    tt =    coo_matrix((data[:, 2], (row_pos, col_pos)),
                        shape=(len(rows), len(cols))).A
    f = interp.RectBivariateSpline(rows,cols,tt)
    return f

class Cell2StationTT():
    def __init__(self,network,station,phase,vel_model,tt):
        self.tt= tt
        self.network = network
        self.station = station
        self.vel_model = vel_model
        self.phase = phase

    def tt2df(self):
        columns = ['src_z', 'src_lat', 'src_lon', 
                    f'{self.network}.{self.station}.{self.phase}.time']
        src = ["src_z","src_lat","src_lon"]
        with cf.ThreadPoolExecutor() as executor:
            exe = executor.map(np.ravel,np.meshgrid(*map(np.arange,  self.tt.shape), indexing="ij"))
            arr = np.column_stack(list(exe) + [ self.tt.ravel()] )

        # arr = np.column_stack(list(map(np.ravel, np.meshgrid(*map(np.arange,  self.tt.shape), indexing="ij"))) + [ self.tt.ravel()])
        df = pd.DataFrame(arr, columns = columns)
        geo_min_coords = sph2geo(self.vel_model.min_coords) #lat,lon,z
        z,radian_lat,radian_lon = self.vel_model.node_intervals
        geo_node_intervals = np.rad2deg(radian_lat),np.rad2deg(radian_lon),z 

        # df["ev_id"] = df.groupby(by=src).ngroup(ascending=True) 
        df["src_lat"] = round(geo_min_coords[0] - df["src_lat"]*geo_node_intervals[0],4) 
        df["src_lon"] = round(geo_min_coords[1] + df["src_lon"]*geo_node_intervals[1],4)
        df["src_z"] = round(geo_min_coords[2] - df["src_z"]*geo_node_intervals[2],4)

        # df["sta"] = df["sta"].apply(lambda x: self.network_order[x])
        # df["net"] = self.network
        # df["sta"] = self.station
        # df = df.set_index(src)
        df = df.set_index(["src_z","src_lat"])
        df = df.groupby(by=["src_z"])

        # for i in df.__iter__():
        #     print(i)
        return df

class SingleTravelTime():
    def __init__(self,vel_model,network_path):
        """
        Parameters:
        -----------
        vel_model_path: tuple
            (Phase name,H5 path of the model)
        network_path: str
            CSV path of the network geometry.(network,station,longitude,latitude,elevation)
            Each station will be the source and the receptors will be 
            the cells of the grid
        """
        self.phase, self.vel_model_path = vel_model
        self.network_path = network_path

    @property
    def network(self):
        network = pd.read_csv(self.network_path)
        network = network.drop_duplicates(["network", "station"])
        network = network.set_index(["network", "station"])
        network = network.sort_index()
        return network

    @property
    def vel_model(self):
        model = pykonal.fields.read_hdf(self.vel_model_path)
        return model

    def _solver(self,iterrow):
        '''
        iterrow: iterative element
        '''
        index, row = iterrow
        network, station = index
        latitude, longitude, elevation = row[["latitude", "longitude", "elevation"]]
        depth = -elevation 
        model = self.vel_model
        # print(f"{self.phase}-{network}.{station}")
        solver = pykonal.solver.PointSourceSolver(coord_sys=model.coord_sys)
        solver.vv.min_coords = model.min_coords
        solver.vv.node_intervals = model.node_intervals
        solver.vv.npts = model.npts
        solver.vv.values = model.values
        solver.src_loc = pykonal.transformations.geo2sph(np.array([latitude, longitude, depth]))
        solver.solve()

        tt = solver.tt.values
        c2t = Cell2StationTT(network,station,self.phase,model,tt)
        df = c2t.tt2df()

        def main(element):
            _,df = element
            dep = df.index.tolist()[0][0]
            df.index = df.index.get_level_values(1)
            df = df.reset_index()
            data = np.array([df["src_lat"].tolist(),
                            df["src_lon"].tolist(),
                            df[f"{network}.{station}.{self.phase}.time"].tolist()]).T

            g1 = self.hf.create_group(f'/{dep}/{network}.{station}.{self.phase}')
            g1.create_dataset('data', data=data)

        for element in df.__iter__():
            main(element)

        # with  cf.ThreadPoolExecutor() as executor:
        #     executor.map(main,df.__iter__())
        # return solver.tt.values

    def solver(self,out=None,n_processor=6):
        """
        Parameters
        ----------
        src: str
            'Station' as source or 'CellOfGrid' as source
        """
        
        self.out = out
        iterrows = self.network.iterrows() 

        ut.isfile(self.out)
        self.hf = h5py.File(self.out, 'w')
        # self.hf = h5py.File(self.out, 'a')
        for iterrow in iterrows:
            (network, station), loc = iterrow
            loc = loc.to_numpy()
            ut.printlog("info",f"{self.phase}-{network}.{station}","running")
            a = time.time()
            self._solver(iterrow)
            b = time.time()
            ut.printlog("info",f"{self.phase}-{network}.{station}",
                        f"ok-time: {round(b-a,2)} seconds")


            g2 = self.hf.create_group(f'info/{network}.{station}')
            g2.create_dataset('geometry', data=loc)

        # self.hf.close()
        # with cf.ThreadPoolExecutor(max_workers=n_processor) as executor:
        #     values = np.array(list(executor.map(self._solver,iterrows)))

class TravelTime():
    def __init__(self, vel_models,network,tt_paths):
        self.vel_models = vel_models
        self.network = network
        self.tt_paths = tt_paths

    def _solver(self,tt_path):
        phase, tt_path = tt_path
        TT = SingleTravelTime(vel_model=(phase.upper(),
                                        self.vel_models[phase.upper()]),
                            network_path= self.network)
        # ut.isfile(tt_path)
        TT.solver(tt_path)

    def solver(self):
        tt_paths = list(zip(list(self.tt_paths.keys()),
                            list(self.tt_paths.values())))
        ut.printlog("info",f"Traveltime.solver","running")
        a = time.time()
        with cf.ThreadPoolExecutor() as executor:
            executor.map(self._solver,tt_paths)
        b = time.time()
        ut.printlog("info",f"Traveltime.solver",        
                    f"Total time: {round(b-a,2)} seconds")

class ReadSingleTraveltime():
    """
    Util class to understand the TravelTime information
    """

    def __init__(self,tt_path):
        """
        Parameters:
        -----------
        tts: dict
            key: str: "Name of the traveltime wave"
            value: str: csv travel time path 
                
        """
        self.tt_path = tt_path
        self.hf = h5py.File(tt_path,'r')
        self.depths = list(self.hf.keys())
        self.depths.remove("info")

    @property
    def network_geometry(self):
        info = self.hf.get("info")
        stations = list(info.keys())
        locs = []
        for station in stations:
            net,sta = station.split(".")
            ind = np.array([net,sta])
            loc = np.array(info.get(station+"/geometry"))
            loc = np.concatenate((ind,loc))
            locs.append(loc)
        locs = np.array(locs)

        df = pd.DataFrame(locs,columns=["network","station",
                                            "longitude",
                                        "latitude","elevation"])
        df[["longitude","latitude",
            "elevation"]] = df[["longitude","latitude",
                                "elevation"]].apply(pd.to_numeric)

        return df

    @property
    def stations(self):
        dep0 = self.depths[0]
        g1 = self.hf.get(dep0)
        g1_items = list(g1.keys())
        return g1_items

    def _get_values(self,dep,phase,network,station):
        dep_str = str(round(float(dep),1))
        netstaphase = ".".join((network,station,phase))
        g1 = self.hf.get(dep_str)
        g1_items = list(g1.items())
        dataset = np.array(g1.get(f"{netstaphase}/data"))
        return dataset

    def _get_traveltime(self,dep,lat,lon,
                        phase,network,station,
                        origin=None):

        data = self._get_values(dep,phase,network,station)


        if len(data.shape) == 0:
            print(f"There is not data in {(dep,phase,network,station)}")
            return

        value = interp2d(data)(lat,lon)
        value = float(np.squeeze(value))

        return value

    def get_traveltimes(self,dep,lat,lon,phase,
                      networks,stations,origin=None,
                      n_false_picks=None ):

        tts = []
        for network in networks:
            for station in stations:

                netstaphase = ".".join((network,station,phase))
                if netstaphase in self.stations:
                    try:
                        tt = self._get_traveltime(dep,lat,lon,
                                            phase,network,station,
                                            origin)
                        if tt == None:
                            continue
                    except Exception as e:
                        print(f"{netstaphase} is not available",e)
                        continue
                    tts.append([network,station,tt])

        df = pd.DataFrame(tts,columns=["network","station",f"{phase.upper()}_tt"])
        

        def add_seconds(row):
            phase_time = row[f"{phase.upper()}_tt"]
            veracity = row[f"{phase.upper()}_veracity"]

            if veracity == 0:
                delta = np.random.uniform(3,15)
                if delta > phase_time:
                    phase_time += delta
                else:
                    phase_time += delta* np.random.choice([-1,1])
            else:
                pass

            return phase_time


        if n_false_picks != None:
            if len(df) > 6: #less than 6 picks don't select false picks
                nums = np.random.choice([0, 1], size=len(df), 
                                    p=[n_false_picks, 1-n_false_picks])
                df[f"{phase.upper()}_veracity"] = nums
                df[f"{phase.upper()}_tt"] = df.apply(add_seconds, axis = 1)
            else:
                df[f"{phase.upper()}_veracity"] = 1
        else:
            df[f"{phase.upper()}_veracity"] = 1

        cols = df.columns
        if origin != None:
            df["origin"] = origin._get_datetime()
            df[f"{phase.upper()}_tt"] = pd.to_timedelta(df[f"{phase.upper()}_tt"],
                                                        unit="seconds") +\
                                        df["origin"]

        return df[cols]

class ReadTraveltime():
    def __init__(self,tt_paths):
        self.tt_paths = tt_paths

    @property
    def network_geometry(self):
        net_geom = {}
        for key,value in self.tt_paths.items():
            tt = ReadSingleTraveltime(value)
            net_geom[key] = tt.network_geometry
        return net_geom

    @property
    def stations(self):
        stations = {}
        for key,value in self.tt_paths.items():
            tt = ReadSingleTraveltime(value)
            stations[key] = tt.stations
        return stations

    @property
    def depths(self):
        depths = {}
        for key,value in self.tt_paths.items():
            tt = ReadSingleTraveltime(value)
            depths[key] = tt.depths
        return depths

    def get_traveltimes(self,dep,lat,lon,
                      networks,stations,
                      origin=None,n_false_picks=None ):
        dfs = []
        for key,value in self.tt_paths.items():
            tt = ReadSingleTraveltime(value)
            df = tt.get_traveltimes(dep,lat,lon,key,
                      networks,stations,origin,n_false_picks)
            dfs.append(df)
        
        df = pd.merge(dfs[0],dfs[1],how="left")
        return df

class Client(ReadTraveltime):
    def __init__(self,tt_paths) -> None:
        super().__init__(tt_paths)

    @property
    def station_availability(self):
        networks = []
        stations = []
        for key,value in self.stations.items():
            for val in value:
                net,sta,_ = val.split(".")
                networks.append(net)
                stations.append(sta)
        return {"networks":set(networks),
                "stations":set(stations)}

    # def _get_events(params):

    def get_event(self,minlatitude, maxlatitude, 
        minlongitude, maxlongitude, mindepth,maxdepth,
        n_stations=None,
        n_false_picks = None,
        station_choice="epicenter_distance",
        origin=None):

        lat = round(np.random.uniform(minlatitude, maxlatitude),2)
        lon = round(np.random.uniform(minlongitude, maxlongitude),2)
        depth = np.random.randint(mindepth, maxdepth)
        # lat,lon,depth =7.42,-74.9,198


        # print(lat ,lon,depth)

        sta_aval = self.station_availability
        df = self.get_traveltimes(depth,lat,lon,sta_aval["networks"],
                                    sta_aval["stations"],origin=origin,
                                    n_false_picks=n_false_picks)

        sset = [ key+"_tt" for key in self.stations.keys()]
        veracity_sset =  [ key+"_veracity" for key in self.stations.keys()]
        df.dropna(subset=sset,inplace=True)

        
        if station_choice == "epicenter_distance":
            metadata = list(self.network_geometry.values())[0]
            df = pd.merge(df,metadata,on=["network","station"])
            gps2da_gen = lambda x: gps2dist_azimuth(lat,lon,
                                        x.latitude,x.longitude)
            df[["r","az","baz"]] =  pd.DataFrame(df.apply(gps2da_gen,axis=1).tolist())
            df["r"] =  df["r"]/1000 #km
            df["weigths"] = 1/df["r"]**2

            weights="weigths"

        else:
            weights=None

        if n_stations != None:

            if isinstance(n_stations, tuple):
                n_stations = np.random.randint(*n_stations)

            if n_stations > len(df):
                n_stations = len(df)
                # raise Exception(f"n_stations={n_stations} exceded the number of available stations")

            df = df.sample(n=n_stations,weights=weights,ignore_index=True)
        # print(df)



        df[["src_lat","src_lon","src_depth"]] = lat ,lon,depth
        df = df.sort_values(by=sset,ignore_index=True)

        cols = ["src_lat","src_lon","src_depth"] + ["network","station"] +\
                ["latitude","longitude","elevation"]+ sset + veracity_sset 
        df = df[cols]
        return df


    def get_events(self,minlatitude, maxlatitude, 
        minlongitude, maxlongitude, mindepth,maxdepth,
        n_events,n_stations=None, n_false_picks=None,
        station_choice="epicenter_distance",
        origin=None,
        max_workers=None):

        def get_only_one_event(id):
            print("Event", id+1)
            df = self.get_event(minlatitude, maxlatitude, 
                                minlongitude, maxlongitude, mindepth,maxdepth,
                                n_stations,station_choice=station_choice,
                                n_false_picks=n_false_picks,
                                origin=origin)
            cols = list(df.columns)
            df["id"] = id

            cols = ["id"] + cols
            df = df[cols]
            return df

        ids = np.arange(0,n_events) 
        with cf.ThreadPoolExecutor(max_workers) as executor:
            events = list(executor.map(get_only_one_event,ids))

        df = pd.concat(events)
        return df


if __name__ == "__main__":
    import time
    from obspy import UTCDateTime
    vel_models = {"P":"/home/emmanuel/results/PEPA/test_col/vel/P_VelModel.h5",
            "S":"/home/emmanuel/results/PEPA/test_col/vel/S_VelModel.h5" }  #modelos de velocidad creados
    # network = "/home/emmanuel/PEPA/data/Colombia.csv" # geometria de red
    # network = "/home/emmanuel/PEPA/data/Colombia_col.csv" # geometria de red
    tt_paths = {"P":"/home/emmanuel/results/PEPA/test_col/tt/P_tt.h5",
                "S":"/home/emmanuel/results/PEPA/test_col/tt/S_tt.h5" } #donde se guardaran archivos necesarios


    client = Client(tt_paths)
    events = client.get_events(7.32,7.45,-74.76,-74.9)
    print(events)
    # tt = ReadTraveltime(tt_paths)
    # print(tt.stations)
    # ## LOS SANTOS
    # data = tt.get_traveltimes(45,7.32,-74.86,
    #                             ["CM"],["ZAR","PTB","HEL"],
    #                             origin=UTCDateTime("2021-10-29 17:22:30") )
    # print(data)