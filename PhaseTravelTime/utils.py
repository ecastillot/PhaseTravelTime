import pandas as pd
import os 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from obspy.core.inventory.inventory import read_inventory
import logging
import time
from obspy import UTCDateTime
from obspy.core.event.catalog import read_events
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
                   datefmt='%m-%d %H:%M') 



def printlog(levelname,name,msg):
    """
    Parameters:
    -----------
    author: str
    """
    logger = logging.getLogger(name)
    if levelname in ("info","information","INFO","Info","INFORMATION"):
        logger.info(msg)
    elif levelname in ("warning","Warning","WARNING"):
        logger.warning(msg)
    elif levelname in ("error","ERROR"):
        logger.error(msg)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def sfile_name(date):
    year = str(date.year).zfill(4)
    month = str(date.month).zfill(2)
    day = str(date.day).zfill(2)
    hour = str(date.hour).zfill(2)
    minute = str(date.minute).zfill(2)
    second = str(date.second).zfill(2)
    sfile_name = day+'-'+hour+minute+'-'+second+'L.S'+year+month
    return sfile_name

def isfile(filepath):
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
                return False
            elif inp.upper() == "N":
                return  True
            else:
                pass
    else:
        return False

def make_station(xml_path):
    inv = read_inventory(xml_path)
    networks = []
    stations = []
    longitudes = []
    latitudes = []
    elevations = []
    for net in inv:
        for sta in net:
            networks.append(net.code) 
            stations.append(sta.code) 
            longitudes.append(sta.longitude) 
            latitudes.append(sta.latitude) 
            elevations.append(sta.elevation/1000) 
    stationdict = {"network":networks,"station":stations,
                    "longitude":longitudes,"latitude":latitudes,
                    "elevation":elevations}
    df = pd.DataFrame(stationdict)
    return df

def make_stationfile(xml_list):
    dfs = []
    for xml_path in xml_list:
        df = make_station(xml_path)
        dfs.append(df)
    
    df = pd.concat(dfs)
    return df

def change_xml_version(ev_file,new_version="0.9"):
    lines = open(ev_file).readlines()
    lines[1] = f'<seiscomp xmlns="http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/{new_version}" version="{new_version}">\n'
    with open(ev_file, 'w') as f:
        f.write(''.join(lines))

def get_picks(seiscomp_file, from_format="SC3ML",version="0.10"):
    """
    parameters
    ----------
    seiscom_file : str
        path of the sc3ml seiscomp file that contains only one event
    export: str (deault : None)
        Exports the picks from the respective event to 'csv' or 'excel' file. None don't create any file, then only returns.
    returns
    -------
    appended_events : list
        It's a list of Pandas DataFrames where each dataframe contains all information about the picks from the respective event
    """
    if from_format == "SC3ML":
        if version not in ["0.5", "0.6", "0.7", "0.8", "0.9", "0.10"]:
            change_xml_version(seiscomp_file,new_version="0.10")

    catalog = read_events(seiscomp_file, format = "SC3ML" )
    event_list = catalog.events 

    appended_events = []
    for event in event_list:
        ev_id = os.path.basename(str(event.resource_id))
        appended_picks = []
        for pick in event.picks:
            appended_items = []
            for pick_key,pick_value in pick.items():
                if pick_key in ('time_errors','waveform_id',
                                'horizontal_slowness_errors',
                                'backazimuth_errors',
                                'creation_info'):

                    if pick_value != None:
                        for new_key, new_item in pick_value.items():
                            appended_items.append((new_key, new_item))

                elif pick_key in ('comments'):
                    if not pick_value:
                        appended_items.append((pick_key,None))
                    else:
                        appended_items.append((pick_key,pick_value))
                else:
                    appended_items.append((pick_key,pick_value))
                
            pick_dict = dict(appended_items)
            pick_df = pd.DataFrame( pick_dict.values(), 
                                    index= pick_dict.keys() ).T
            pick_df["id"] = ev_id
            pick_df = pick_df[["id","network_code","station_code","phase_hint","time"]]
            appended_picks.append(pick_df)

        df_picks = pd.concat(appended_picks,ignore_index=True,sort=False)
        appended_events.append(df_picks)

    df_picks = pd.concat(appended_picks,ignore_index=True,sort=False)

    p_df_picks = df_picks[df_picks["phase_hint"]=="P"]
    s_df_picks = df_picks[df_picks["phase_hint"]=="S"]
    df_picks = pd.merge(p_df_picks,s_df_picks,how="left",on=["id","network_code","station_code"],suffixes=("_P","_S"))
    df_picks = df_picks.rename(columns={"network_code":"network","station_code":"station",
                            "time_P":"P_tt","time_S":"S_tt"})
    df_picks = df_picks[["id","network","station","P_tt","S_tt"]]
    return df_picks

def plot_station(st,station,data,sgc_picks,ax=None):

    tr = st.select(station=station,channel="*Z")[0]

    Ppick = UTCDateTime(data[data["station"]==station].reset_index().P_tt[0])-tr.stats.starttime 
    Spick = UTCDateTime(data[data["station"]==station].reset_index().S_tt[0])-tr.stats.starttime 

    sgc_Ppick = UTCDateTime(sgc_picks[sgc_picks["station"]==station].reset_index().P_tt[0])-tr.stats.starttime 
    sgc_Spick = UTCDateTime(sgc_picks[sgc_picks["station"]==station].reset_index().S_tt[0])-tr.stats.starttime 

    if ax ==None:
        fig,ax = plt.subplots()

    ax.plot(tr.data, 'k')
    ymin, ymax = ax.get_ylim()
    ax.vlines(Ppick *100, ymin, ymax, color='r', linewidth=2)
    ax.vlines(Spick*100, ymin, ymax, color='b', linewidth=2)
    ax.vlines(sgc_Ppick *100, ymin, ymax, color='r', linewidth=2,linestyles="dashed")
    ax.vlines(sgc_Spick*100, ymin, ymax, color='b', linewidth=2,linestyles="dashed")
    ax.set_xlabel("samples")
    ax.set_ylabel("Counts")
    ax.set_title(f"{tr.stats.station}  {tr.stats.starttime }")
    ax.axis('tight')
    if ax == None:
        ax.legend(["P-phase","S-phase" ,"sgc_P-phase","sgc_S-phase"], 
                    fontsize='medium',
                    loc ="lower left")
        
        
        
        return ax

    return ax

def plot_stations(st,data,sgc_picks,**kwargs):
    true_stations = list(set([tr.stats.station for tr in st]))
    data_stations = data["station"].to_list()
    stations = [value for value in data_stations if value in true_stations]

    fig,axes = plt.subplots(len(stations),1,**kwargs)
    for i,station in enumerate(stations):
        plot_station(st,station,data,sgc_picks,axes[i])
    plt.legend(["data","P-phase","S-phase" ,"manual_P-phase","manual_S-phase"], 
                    fontsize='medium',
                    loc ="lower left")
    return fig    

if __name__ == "__main__":
    Colombia = "/home/emmanuel/Phasepy/data/Colombia.csv"
    CM_path = "/home/emmanuel/AIpicker/others/public_CM.xml"
    YU_path = "/home/emmanuel/AIpicker/others/YU.xml"
    xml_list = [CM_path,YU_path]
    df = make_stationfile(xml_list)
    df.to_csv(Colombia,index=False)


    # pl = Printlog("info","test","hola")
    # printlog(pl)
