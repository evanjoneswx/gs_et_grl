'''
Evan Jones, Florida State University, Department of Earth, Ocean and Atmospheric Science

Function for calculating the Gulf Stream SST gradient Index

Last updated: January 12, 2023
'''

def sst_gradient_index_calc(sst_grad, sst_grad_climatology, MMDD):
    '''
    Calculates Gulf Stream SST gradient Index time series, 
    selecting only during the NATL Hurricane Season
    Inputs:
        sst_grad: Temperature (K)
        sst_grad_climatology: u-component of wind (m/s) (calculated using daily_means.py)
    Output:
        sst_grad_index_hs: Gulf Stream SST gradient Index time series
        only during the NATL hurricane season
    '''
    
    import numpy as np 
    import xarray as xr
    import pandas as pd
    from scipy import signal
    import matplotlib.pyplot as plt
    import proplot as plot

    # resample the original to get the resampled daily value of the SST gradient
    sst_grad = sst_grad.resample(time='1D').mean(skipna=True)
    time_list = sst_grad.time

    MMDD = sst_grad_climatology.MMDD

    # Lats and lons of the locations of the GS SST gradient index
    lats = np.array([31, 32, 33, 34]) 
    lons = np.array([-80.5, -79.75, -78, -76.5]) % 360 

    # loop through the files and select the lats/lons of interest for the index
    mean_all = []
    for current_time in time_list:
        monthday = int(current_time.dt.strftime('%m%d'))
        sst_grad_list = []
        for lat, lon in zip(lats,lons):
            sst_grad_indiv = sst_grad.sel(time=current_time,latitude=lat,longitude=lon).values
            sst_grad_day = sst_grad_climatology.sel(MMDD=monthday,latitude=lat,longitude=lon).values
            sst_grad_anom = sst_grad_indiv - sst_grad_day
            sst_grad_list.append(sst_grad_anom)
        sst_grad_array = np.array(sst_grad_list)
        # take the mean at each time of the four locations selected
        mean_sst_grad = np.nanmean(sst_grad_array)
        mean_all.append(mean_sst_grad)

    sst_grad_index = signal.detrend(np.array(mean_all))
    sst_grad_index_xr = xr.DataArray(sst_grad_index,dims=['time'],coords=[time_list],name='gs_index')
    sst_grad_index_hs = sst_grad_index_xr.sel(time=(ds.time.dt.month==6) | (ds.time.dt.month==7) | (ds.time.dt.month==8) | (ds.time.dt.month==9) | (ds.time.dt.month==10) | (ds.time.dt.month==11))
    return sst_grad_index_hs