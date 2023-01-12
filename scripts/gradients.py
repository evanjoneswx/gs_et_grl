'''
Evan Jones, Florida State University, Department of Earth, Ocean and Atmospheric Science

Function for calculating sea surface temperature gradients
and sensible heat flux gradients (as described by Parfitt et al. 2016):

Last updated: January 12, 2023
'''

def sst_gradient_calc(sst):
    '''
    Calculates adiabatic frontogenesis at all pressure levels
    Inputs:
        sst: Sea surface temperature (K)
    Output:
        sst_grad_mag: magnitude of SST gradient (K/m)
    '''
    import numpy as np 
    import pandas as pd
    import xarray as xr
    import metpy.calc 
    from metpy.units import units

    # calculate the grid deltas
    lat = sst.latitude.values
    lon = sst.longitude.values
    dx, dy = metpy.calc.lat_lon_grid_deltas(lon,lat)

    # calculate the SST gradient
    sst_gradient = metpy.calc.gradient(sst * units.K,deltas=(dy,dx) * units.meter)
    sst_grad_mag = np.sqrt((sst_gradient[0]**2) + (sst_gradient[1]**2))
    sst_grad_mag = xr.DataArray(sst_grad_mag)
    return sst_grad_mag
    # also save the components
    return sst_grad

def shf_gradient_calc(sst, shf, sst_grad_mag, sst_grad):
    '''
    Calculates the sensible heat flux gradient:
    Inputs:
        sst: Sea surface temperature (K)
        shf: Sensible heat flux (W/m^2)
        sst_grad_mag: Magnitude of SST gradient (K/m)
        sst_grad: x and y components of SST gradient (K/m)
    Output:
        shf_grad: Sensible heat flux gradient (W/m^2/m)
    '''
    import numpy as np 
    import xarray as xr
    import pandas as pd
    import metpy.calc 
    from metpy.units import units

    # calculate the normal vector, n
    sst_grad_x = sst_grad[1] / sst_grad_mag
    sst_grad_y = sst_grad[0] / sst_grad_mag
    n = xr.concat([sst_grad_y,sst_grad_x],dim='dir')

    # calculate the sensible heat flux gradient in x and y coordinates
    shf_grad = metpy.calc.gradient(shf_sel, deltas=(dy,dx) * units.meter)
    shf_grad_x = xr.DataArray(shf_grad[1],dims=['latitude','longitude'],coords=[latitude,longitude],name='shf')
    shf_grad_y = xr.DataArray(shf_grad[0],dims=['latitude','longitude'],coords=[latitude,longitude],name='shf')
    shf_grad_xr = xr.concat([shf_grad_y,shf_grad_x],dim='dir')

    # calculate the cross-frontal sensible heat flux gradient
    shf_grad = xr.dot(n,shf_grad_xr,dims='dir')
    return shf_grad