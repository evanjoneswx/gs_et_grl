'''
Evan Jones, Florida State University, Department of Earth, Ocean and Atmospheric Science

Function for calculating adiabatic (as described by Bluestein, 1993) 
and diabatic frontogeneis (as described by Reeder et al., 2021):

Last updated: January 12, 2023
'''
def adiabatic_fgen_calc(temp, uwind, vwind, level):
    '''
    Calculates adiabatic frontogenesis at all pressure levels
    Inputs:
        temp: Temperature (K)
        uwind: u-component of wind (m/s)
        vwind: v-component of wind (m/s)
        level: 1D array of pressure levels (hPa)
    Output:
        ad_fgen: adiabatic frontogenesis (K/m/s)
    '''
    import numpy as np 
    import pandas as pd
    import xarray as xr
    import metpy.calc 
    from metpy.units import units

    # calculate potential temperature using metpy
    pot_temp = metpy.calc.potential_temperature(level * units.mbar,temp)
    pot_temp = pot_temp.metpy.assign_crs(grid_mapping_name='latitude_longitude',earth_radius=6371229.0)

    # calculate adiabatic frontogenesis
    ad_fgen = metpy.calc.frontogenesis(pot_temp,uwind,vwind)
    return ad_fgen

def diabatic_fgen_calc(timestamp, temp, uwind, vwind, level, longitude, latitude):
    '''
    Calculates diabatic frontogenesis at at specified pressure levels
    Inputs:
        timestamp: Time of calculation
        temp: Temperature (K) for time before, timestamp, and time after
        uwind: u-component of wind (m/s)
        vwind: v-component of wind (m/s)
        level: level: 1D array of pressure levels (hPa) 
        Note: If calculating for the surface, use an array of surface pressure
    Output:
        diab_fgen: diabatic frontogenesis (K/m/s)
    '''
    import numpy as np 
    import pandas as pd
    import xarray as xr
    import metpy.calc 
    from metpy.units import units

    # calculate potential temperature 
    pot_temp = metpy.calc.potential_temperature(level * units.mbar,temp)
    pot_temp = pot_temp.metpy.assign_crs(grid_mapping_name='latitude_longitude',earth_radius=6371229.0)
    # calculate grid deltas
    dx, dy = metpy.calc.lat_lon_grid_deltas(longitude,latitude)
    dy = -1*dy

    # calculation advection of pot temp by u and v winds
    adv = metpy.calc.advection(pot_temp.sel(time=timestamp),uwind,vwind) # units of K/s
    # calculate time tendency of potential temperature
    dtheta_dt = metpy.calc.first_derivative(pot_temp,axis='time').sel(time=timestamp)
    # diabatic heating rate
    theta_dot = dtheta_dt + adv
    # normal vector in order y, x
    pot_temp_grad = metpy.calc.gradient(pot_temp.sel(time=timestamp), deltas=(dy,dx) * units.meter)

    # calculate the magnitude of sst gradient
    pot_temp_grad_mag = np.sqrt((pot_temp_grad[0]**2) + (pot_temp_grad[1]**2))
    pot_temp_grad_mag = xr.DataArray(pot_temp_grad_mag,dims=['latitude','longitude'],coords=[latitude,longitude],name='pot_temp_grad_mag')

    # calculate the normal vector, n
    pot_temp_grad_x = pot_temp_grad[1] / pot_temp_grad_mag
    pot_temp_grad_y = pot_temp_grad[0] / pot_temp_grad_mag
    n = xr.concat([pot_temp_grad_y,pot_temp_grad_x],dim='dir')
    # calculate the diabatic heating gradient
    theta_dot_grad = metpy.calc.gradient(theta_dot, deltas=(dy,dx) * units.meter)
    theta_dot_grad_x = xr.DataArray(theta_dot_grad[1],dims=['latitude','longitude'],coords=[latitude,longitude],name='theta_dot')
    theta_dot_grad_y = xr.DataArray(theta_dot_grad[0],dims=['latitude','longitude'],coords=[latitude,longitude],name='theta_dot')
    theta_dot_grad_xr = xr.concat([theta_dot_grad_y,theta_dot_grad_x],dim='dir')
    # flip to ensure the gradient goes from cold to warm
    theta_dot_grad_xr = theta_dot_grad_xr * -1

    # calculate diabatic frontogenesis
    diab_fgen = xr.dot(n,theta_dot_grad_xr,dims='dir')
    return diab_fgen