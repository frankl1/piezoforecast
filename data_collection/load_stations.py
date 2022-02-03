#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load stations with at least 1500 measurements

@author: T. Guyet, Inria
@date: 02/2022
"""


parquet_installed=False
try:
    import pyarrow.parquet as pq
    parquet_installed=True 
except:
    pass
import json
import pandas as pd

import requests

def code_dep():
    code_departement=['01','02','03','04','05','06','07','08','09']
    for i in range(10,20):
        code_departement.append(str(i))
    code_departement+=['2A','2B']
    for i in range (21,69):
        code_departement.append(str(i))
    code_departement+=['69D','69M']
    for i in range (70,95):
        code_departement.append(str(i))
    #we only take the piezometer in the metropolitain area of France
    #code_departement+=['971','972','973','974','975','976','977','978','984','986','987','988','989']
    return code_departement

depts = code_dep()

stations=[]

## On charge les 50 premiers départements !
next_page = 'http://hubeau.eaufrance.fr/api/v1/niveaux_nappes/stations'
params = {'code_departement':','.join(depts[:50]), 'nb_mesures_piezo_min':'1500'}
while next_page is not None:
    response = requests.get(next_page,params=params)
    print(next_page)
    data=response.json()   
    try:
        next_page = data['next']
    except:
        #il y a eu une erreur / ou c'est la dernière page
        print(data)
        break
    #on récupere les données de la page
    stations += data['data']

## On charge les suivants    
next_page = 'http://hubeau.eaufrance.fr/api/v1/niveaux_nappes/stations'
params = {'code_departement':','.join(depts[50:]), 'nb_mesures_piezo_min':'1500'}
while next_page is not None:
    response = requests.get(next_page,params=params)
    print(next_page)
    data=response.json()   
    try:
        next_page = data['next']
    except:
        #il y a eu une erreur / ou c'est la dernière page
        print(data)
        break
    #on récupere les données de la page
    stations += data['data']  
    
#transformation en data.frame pandas
stations=pd.DataFrame(stations)


#Selection d'un sous-ensemble de caractéristiques
stations = stations[['code_bss','x', 'y', 'geometry', 'code_departement','profondeur_investigation', 'altitude_station', 'noms_masse_eau_edl',"codes_bdlisa"]]
stations.columns = ['bss','x', 'y', 'geometry', 'dpt', 'prof', 'alt', 'masse_eau', "codes_bdlisa"]
#recupération du premier code dans la liste des codes de BDLISA
stations.codes_bdlisa=stations.codes_bdlisa.map(lambda x: None if x is None else x[0])

#on rajoute les codes de la BD Lisa
bdlisa = pd.read_csv("bdlisa_simple.csv").drop_duplicates()
stations = stations.merge(bdlisa, left_on="codes_bdlisa", right_on="CodeEH", how="left")


if parquet_installed:
    stations.to_parquet('stations.parquet')

filename = 'stations.csv'
stations.to_csv(filename, index=False)

filename = 'list_bss.csv'
stations['bss'].to_csv(filename, index=False)