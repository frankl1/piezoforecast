## Data collection
This directory 

### pre-requires
- `bdlisa_simple.csv` which is an extract of the BD Lisa to get the characteristics of soils


### steps to create the dataset
1. Execute `load_stations.py' to get a list of piezometers from Hubeau and a file that describes the stations. Yields:
	- `list_bss.csv`
	- `stations.csv`
2. Execute `load_cdsdata.py` to load the climate database. Yields, per year:
	- `rain_XXXX.nc`
	- `total_evaporation_XXXX.nc`
	-> this step requires to get a (free) access to the CDC database and to install the CDS API
	-> 2Gb data
	-> this step may take time due to the response delay of the CDS services

3. Execute `load_dataset.py` to create the dataset. It downloads the time series of piezometers from Hubeau and then merges the dataset with rain and ETP. Yields:
	- `dataset_XXXX_XXXX.csv` 
	-> this step takes time (download each bss independantly)
	



