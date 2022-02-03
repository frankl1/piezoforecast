#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generation des jeux de données

- pré-requis
  * génération de rain_2020.nc à partir du script 'load_cdc.py'
  * generation de la liste de bss d'intérêts

- Chargement des données maillées de température et d'évapotranspiration
  * https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview
  * voir script load_cdsdata.py

  
@author: T. Guyet, Inria
"""

import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
import requests

rep_data="."
annee_debut=2015

date_debut=str(annee_debut)+'-01-01'
date_fin='2021-07-28'

#include_bss=["00471X0095/PZ2013","00487X0015/S1","00755X0006/S1"  ,"00762X0004/S1","00766X0004/S1","01258X0020/S1","01516X0004/S1","01584X0023/LV3","02206X0022/S1","02267X0030/S1","02603X0009/S1","02648X0020/S1","02706X0074/S77-20","03124X0088/F","04398X0002/SONDAG","06505X0080/FORC","07223C0113/S","07476X0029/S"]

#list_bss=pd.read_csv(rep_data+"/list_bss.csv")['x'].to_list()+include_bss
list_bss=pd.read_csv(rep_data+"/list_bss.csv")['bss']
list_bss=list(set(list_bss))

#Base de piézometres avec ses caracteriistiques
stations = pd.read_csv(rep_data+"/stations.csv")

#######################################################"
# Get the measurements of groundwater level
print("Number of BSS time series to load: ", len(list_bss))
levels=[]
for bss in tqdm(list_bss):
    url = 'http://hubeau.eaufrance.fr/api/v1/niveaux_nappes/chroniques'
    params = {'code_bss':bss, 'date_debut_mesure':date_debut, 'date_fin_mesure':date_fin, 'size':5000}
    response = requests.get(url, params=params)
    val=response.json()
    dates = [ val['data'][j]["date_mesure"] for j in range(len(val['data']) )]
    vals = [ val['data'][i]["profondeur_nappe"] for i in range(len(val['data']) )]
    bss=[ bss ]*len(vals)
    levels.append( pd.DataFrame({'t':pd.to_datetime(dates), 'p':vals, 'bss':bss}) )

levels = pd.concat(levels)
levels=levels.rename(columns={"t":"time"}).set_index(['bss','time'])


#######################################################"
#Add exogeneous variables
eto=[]
rain=[]
for year in range(annee_debut,2022):
    print(f"year: {year}")
    #Ouverture du fichier contenant les données de pluie
    ds_rain = xr.open_dataset(rep_data+"/rain_"+str(year)+".nc")
    df = ds_rain.to_dataframe()
    df.reset_index(inplace=True)
    
    
    ds_eto = xr.open_dataset(rep_data+"/total_evaporation_"+str(year)+".nc")
    #aggregate hourly data by day of year
    ds_eto_agg=ds_eto.groupby("time.dayofyear").sum()
    

    for v in tqdm(stations.iterrows()):
        x=v[1]['x']
        y=v[1]['y']
        try:
            l=df[(df.longitude>=x) & (df.longitude<x+0.25) & (df.latitude>=y) & (df.latitude<y+0.25)][['time','tp']].resample('D', on='time').sum()
            l['bss']=v[1]['bss']
            rain.append(l)
        except AttributeError:
            print("error: possible unknown location (%f,%f)"%(x,y))
            
        try:
            l=ds_eto_agg.where( (ds_eto_agg.longitude>=x) & 
                 (ds_eto_agg.longitude<x+0.10) & 
                 (ds_eto_agg.latitude>=y) & 
                 (ds_eto_agg.latitude<y+0.10) , drop=True).to_dataframe()
            l['bss']=v[1]['bss']
            l.reset_index(inplace=True)
            l['time']=(np.asarray(year, dtype='datetime64[Y]')-1970) +(np.asarray(l['dayofyear'], dtype='timedelta64[D]')-1)
            eto.append(l)
        except AttributeError:
            print("error: possible unknown location (%f,%f)"%(x,y))
        
rain=pd.concat(rain)
rain=rain.reset_index().set_index(['bss','time'])

alleto=pd.concat(eto)
alleto=alleto.reset_index().set_index(['bss','time'])[['e']]

dataset=rain.join(alleto).join(levels)
dataset.to_csv(rep_data+"/dataset_"+str(annee_debut)+"_2021.csv")


   
