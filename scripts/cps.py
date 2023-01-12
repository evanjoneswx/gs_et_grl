'''
Evan Jones, Florida State University, Department of Earth, Ocean and Atmospheric Science

Function for calculating the cyclone phase space (as described by Hart 2003) and
associated functions necessary for the calculations.

Last updated: January 12, 2023
'''

def haversine(lat1, lon1, lat2, lon2):  
    '''
    Calculates the Haversine distance on Earth's surface (great circle distance) between two points 
    Inputs: 
        lat1, lon1: reference coordinates to start with (0 360 for longitude)      
        lat2, lon2: coordinates the distance is being calculated to. 
        Note that lat2/lon2 can be 2D meshgrids as well and would return an array of the distances
    Outputs: 
        dist: Haversine distance (km)
    '''
    import numpy as np
    # distance between latitudes 
    # and longitudes 
    dLat = (lat2 - lat1) * np.pi / 180.0
    dLon = (lon2 - lon1) * np.pi / 180.0
    # convert to radians 
    lat1 = (lat1) * np.pi / 180.0
    lat2 = (lat2) * np.pi / 180.0
    # apply formulae 
    a = (pow(np.sin(dLat / 2), 2) + 
            pow(np.sin(dLon / 2), 2) * 
                np.cos(lat1) * np.cos(lat2)) 
    rad = 6371
    c = 2 * np.arcsin(np.sqrt(a)) 
    dist = rad * c 
    return dist 

def bearing(lat1,lon1,lat2,lon2):
    '''
    Calculates the bearing (from true north clockwise) on a gridded dataset between two points
    Inputs: 
        lat1, lon1: reference coordinates to start with (0 360 for longitude)
        lat2, lon2: coordinates the bearing is being calculated to. 
        Note that lat2/lon2 can be 2D meshgrids as well and would return an array of the bearings
    Outputs: 
    ang: Bearing (degrees) 
    '''
    # constants
    d2r = np.pi / 180 # convert from degrees to radians
    r2d = (1/d2r) # convert radians to degrees

    # convert lats/lons from degrees to radians
    lat1r = lat1*d2r
    lon1r = lon1*d2r
    lat2r = lat2*d2r
    lon2r = lon2*d2r

    # compute angle of motion based on the two lats/lons
    ang = r2d*np.arctan2(np.sin((lon2r-lon1r))*np.cos(lat2r), np.cos(lat1r)*np.sin(lat2r) - np.sin(lat1r)*np.cos(lat2r)*np.cos(lon2r-lon1r))

    # convert from -180 180 to 0 360
    ang = ang % 360
    return ang

def sel_subset(array,center_lon,center_lat,search_box):
    '''
    Selects a subset of data relative to a central reference point and specified search area
    Inputs: 
        array: gridded dataset to make the selection from
        center_lon, center_lat: reference center to select from
        search_box: area to select over (rectangle whose width is dimensions of search_box x search_box)
    Outputs: 
        array_subset: Array subset chosen (will be square in shape)
    '''
    # select a subset of lats/lons within search_box / 2 of center
    lon_min = center_lon - (search_box/2)
    lon_max = center_lon + (search_box/2)
    lat_min = center_lat - (search_box/2)
    lat_max = center_lat + (search_box/2)
    if center_lon >= 360 - (search_box/2): # if there is a center_lon > 350
        remainder = 360 - center_lon 
        array_subset_1 = array.sel(longitude=slice(center_lon-(search_box/2),360),latitude=slice(center_lat+(search_box/2),center_lat-(search_box/2))) 
        array_subset_2 = array.sel(longitude=slice(0,(search_box/2) - remainder),latitude=slice(center_lat+(search_box/2),center_lat-(search_box/2))) 
        array_subset = xr.concat([array_subset_1,array_subset_2],dim='longitude')
    elif center_lon < (search_box/2): # less than 7.5deg
        remainder = search_box/2 - center_lon 
        array_subset_1 = array.sel(longitude=slice(0,center_lon+(search_box/2)),latitude=slice(center_lat+(search_box/2),center_lat-(search_box/2)))
        array_subset_2 = array.sel(longitude=slice(360-remainder,360),latitude=slice(center_lat+(search_box/2),center_lat-(search_box/2)))
        array_subset = xr.concat([array_subset_1,array_subset_2],dim='longitude')
    else:
        array_subset = array.sel(longitude=slice(lon_min,lon_max),latitude=slice(lat_max,lat_min))

    return array_subset

def b_calcs(height,center_lon,center_lat,ang):
    '''
    Calculates the thickness asymmetry across a cyclone for use within the cyclone phase space (CPS).
    Inputs: 
        height: 3D array of geopotential height values in lat/lon (m)
        center_lon, center_lat: Longitude and latitude of TC center (degrees)
        ang: Storm motion bearing (degrees)
    Outputs: 
        B: Thickness asymmetry value (B) for a TC for analysis within the CPS (one value in m)
    '''
    
    import xarray as xr
    import numpy as np
    # constants
    d2r = np.pi / 180 # convert from degrees to radians
    h = 1.0 # h needs to be -1 if in SH

    # select the data to look at
    height_600 = height.sel(level=600)
    height_900 = height.sel(level=900)
    thickness = height_600-height_900

    # make meshgrids of the selected thickness, longitude and latitude
    thickness_subset, lon2d, lat2d = sel_subset(thickness,center_lon,center_lat)
    
    # find the bearing between the center and each lat/lon in the array
    angles_all = bearing(center_lat,center_lon,lat2d,lon2d)

    # find the 500-km radius for Z calculations and only use points inside 500-km radius
    d = haversine(center_lat, center_lon, lat2d, lon2d)
    Zl = xr.where(d < tc_rad,thickness_subset,np.nan) # set points outside 500-km radius to nans
    Zr = xr.where(d < tc_rad,thickness_subset,np.nan) # set points outside 500-km radius to nans
    # set values along great circle line to nans (since they wouldn't be either in the left or right hemisphere of the circle technically)
    Zl = xr.where(angles_all == ang,np.nan,Zl)
    Zr = xr.where(angles_all == ang,np.nan,Zr)
    # for storm motion angles in quadrants 1 & 2 (NE and SE)
    if ang >= 0 and ang < 180:
        Zl = xr.where((angles_all < ang) | (angles_all > ang+180), Zl, np.nan)
        Zr = xr.where((angles_all > ang) & (angles_all < ang + 180), Zr, np.nan)
    # for storm motion angles in quadrants 3 & 4 (NW and SW)
    elif ang >= 180 and ang < 360:
        Zl = xr.where((angles_all > ang - 180) & (angles_all < ang), Zl, np.nan)
        Zr = xr.where((angles_all > ang) | (angles_all < ang - 180), Zr, np.nan)

    # make array for calculating the weighted average
    weighted_lat = np.cos(np.deg2rad(lat2d))
    # only have weights where there are valid values
    Zr_weights = xr.where(np.isnan(Zr),np.nan,weighted_lat)
    Zl_weights = xr.where(np.isnan(Zl),np.nan,weighted_lat)
    # get the weighted value of Zr and Zl
    weighted_Zr = Zr*Zr_weights
    weighted_Zl = Zl*Zl_weights
    # calculate the weighted sum of weighted Zr and weighted Zl
    Zr_sum = weighted_Zr.sum(dim=('latitude','longitude'))
    Zl_sum = weighted_Zl.sum(dim=('latitude','longitude'))
    # sum the weights for dividing by in weighted mean for each side
    Zr_weights_sum = Zr_weights.sum(dim=('latitude','longitude'))
    Zl_weights_sum = Zl_weights.sum(dim=('latitude','longitude'))
    # calculate the weighted average of each side
    Br_weighted = Zr_sum / Zr_weights_sum
    Bl_weighted = Zl_sum / Zl_weights_sum
    # calculate B
    if center_lat > 0: # h parameter for northern hemisphere
        h = 1
    else: # h parameter for southern hemisphere
        h = -1
    B = h * (Br_weighted - Bl_weighted)

    return B


def calc_VltVut(height_all_levs,center_lon,center_lat,lev):
    '''
    Calculates the lower tropospheric and upper tropospheric thermal wind for a cyclone
    Inputs: 
        height_all_levs: 3D array of geopotential height values in lat/lon (m)
        center_lon, center_lat: Longitude and latitude of TC center (degrees)
        lev: 1D array of level values based on height_all_levs (hPa)
    Outputs: 
        Vut: Upper tropospheric thermal wind 
        Vlt: Lower tropospheric thermal wind 
    '''
    import xarray as xr
    import numpy as np
    # constants 
    d2r = np.pi / 180 # convert from degrees to radians
    search_box = 15
    tc_rad = 500
    res = 0.25 # resolution of data
    h = 1.0 # h needs to be -1 if in SH

    # make meshgrids of the selected heights, longitude and latitude
    height_subset, lon2d, lat2d = sel_subset(height_all_levs,center_lon,center_lat)

    lon3d = np.repeat(lon2d[np.newaxis,:,:],lev.size,axis=0)
    lat3d = np.repeat(lat2d[np.newaxis,:,:],lev.size,axis=0)
    # find 500-km radius for thickness calculations
    d = haversine(center_lat, center_lon, lat3d, lon3d)
    thickness_500km = xr.where(d < tc_rad,height_subset,np.nan) # set points outside 500-km radius to nans
    dZ = thickness_500km.max(dim=['latitude','longitude']) - thickness_500km.min(dim=['latitude','longitude'])
    # select slices between 300-600 and between 600-900
    dZu = dZ.sel(level=slice(300,600))
    dZl = dZ.sel(level=(650,700,750,800,850,900))
    # take the natural log of each pressure level
    lnpu = np.log(lev.sel(level=slice(300,600)))
    lnpl = np.log(lev.sel(level=(650,700,750,800,850,900))) 
    # Compute thermal wind using linear regressions
    Vut, bupper = linear_regression(lnpu,dZu)
    Vlt, blower = linear_regression(lnpl,dZl)

    return Vut, Vlt

def weighted_mean(array):
    '''
    Calculates the 24-hr running mean of array of values (used here for smoothing the CPS parameters 
    Inputs: 
        array: Array of values to smooth over 
        (needs to be in 6-hourly intervals for this specific smoothing function)
    Outputs: 
        array_running_avg: 24-hr running mean (smoothed) values of inputted array
    '''
    import numpy as np
    array = np.array(array)
    array_running_avg = array[0:2]
    for i in range(2,len(array)-2):
        array_5_times = array[i-2:i+3]
        array_avg = np.sum(array_5_times) / 5
        array_running_avg = np.append(array_running_avg,array_avg)

    array_end = array[len(array)-2:len(array)]
    array_running_avg = np.append(array_running_avg,array_end)
    return array_running_avg