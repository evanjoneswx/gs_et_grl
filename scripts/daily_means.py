'''
Evan Jones, Florida State University, Department of Earth, Ocean and Atmospheric Science

Function for calculating daily means for removing the seasonal cycle of data

Last updated: January 12, 2023
'''


import numpy as np 
import xarray as xr
import pandas as pd
from scipy import signal

ds = xr.open_mfdataset('/mars/parfitt/era5/forecasts/RAD/lhf_shf/gradients/global/*_shf_grad.nc')
sst = ds.sshf
time = ds.time
ds.close()

shf_grad = sst.sel(longitude=slice(-85 % 360, -20 % 360),latitude=slice(60,20))
latitude = shf_grad.latitude
longitude = shf_grad.longitude

def daily_means_calc(var):
    '''
    Calculates the daily mean of a particular variable
    Inputs: 
        var: variable over which to calculate the daily mean (dimensions time, lat, lon)  
    Outputs: 
        daily_mean: Daily mean values of the quantity
    '''
    import numpy as np 
    import xarray as xr
    import pandas as pd
    from scipy import signal
    # resample to a signle daily value
    daily_resampled = var.resample(time='1D').mean(skipna=True)
    
    # detrend the time series
    original_mean = daily_resampled.mean(dim='time').values
    daily_resampled = xr.where(np.isnan(daily_resampled),-100,daily_resampled)
    detrended_var = signal.detrend(daily_resampled,axis=0,type='linear')
    detrended_var = xr.DataArray(detrended_var,  dims=['time','latitude','longitude'], coords=[daily_resampled.time,daily_resampled.latitude,daily_resampled.longitude], name='detrended_var')
    detrended_var = xr.where(detrended_var==-100,np.nan,detrended_var)
    daily_resampled = detrended_var + original_mean
    
    # create data frames to take daily means over
    time3D, lat3D, lon3D = xr.broadcast(daily_resampled.time, daily_resampled.latitude, daily_resampled.longitude)
    df = pd.DataFrame({'VAR':np.ravel(daily_resampled.values),
                    'time':pd.to_datetime(np.ravel(time3D.values)),
                    'latitude':np.ravel(lat3D.values),
                    'longitude':np.ravel(lon3D.values)})
    df['MMDD'] = df.time.dt.strftime('%m%d') #add month and day column
    df = df.drop(columns='time')
    # take the means over the same data for the total time series
    df_mean = df.groupby(['MMDD','latitude','longitude']).mean().reset_index()
    del df, time3D, lat3D, lon3D
    df_mean = df_mean.sort_values(['MMDD', 'latitude'], ascending=(True, False))
    df_mean = df_mean.reset_index(drop=True)

    # reshape back into a Data Array
    monthday_mean_sst = np.reshape(df_mean.VAR.values, (366,len(latitude),len(longitude)))
    monthday = pd.date_range('2020-01-01','2020-12-31',freq='D').strftime('%m%d').astype(int)
    daily_mean = xr.DataArray(np.reshape(monthday_mean_sst, (366,len(latitude),len(longitude))), dims=['MMDD','latitude','longitude'], coords=[monthday,latitude,longitude], name='VAR_monthday_mean')
    return daily_mean