# stacking.py
# module: vespa.stacking
# Various stacking functions for seismic data

from vespa.utils import get_station_coordinates
import numpy as np


def get_shifts(st, s, baz):
    '''
    Calculates the shifts (as an integer number of samples in the time series) for every station in a stream of time series seismograms for a slowness vector of given magnitude and backazimuth.
    
    The shift is that which needs to be applied in order to align an arrival (arriving with slowness s and backazimuth baz) with the same arrival at the array reference point (the location of the station that makes up the first trace in the stream).

    Parameters
    ----------
    st : ObsPy Stream object
        Stream of SAC format seismograms for the seismic array, length K = no. of stations in array
    s  : float
        Magnitude of slowness vector, in s / km
    baz : float
        Backazimuth of slowness vector, (i.e. angle from North back to epicentre of event)

    Returns
    -------
    shifts : list
        List of integer delays at each station in the array, also length K
    '''
    theta = [] # Angular position of each station, measured clockwise from North
    r = [] # Distance of each station

    # First station is reference point, so has zero position vector
    theta.append(0.0)
    r.append(0.0)

    geometry = get_station_coordinates(st)/1000. # in km

    # For each station, get distance from array reference point (first station), and the angular displacement clockwise from north
    for station in geometry[1:]:
        r_x = station[0] # x-component of position vector
        r_y = station[1] # y-component of position vector

        # theta is angle c/w from North to position vector of station; need to compute diffently for each quadrant
        if r_x >= 0 and r_y >= 0:
            theta.append(np.degrees(np.arctan(r_x/r_y)))
        elif r_x > 0 and r_y < 0:
            theta.append(180 + np.degrees(np.arctan(r_x/r_y)))
        elif r_x <= 0 and r_y <= 0:
            theta.append(180 + np.degrees(np.arctan(r_x/r_y)))
        else:
            theta.append(360 + np.degrees(np.arctan(r_x/r_y)))

        r.append(np.sqrt(r_x**2 + r_y**2))

    # Find angle between station position vector and slowness vector in order to compute dot product

    # Angle between slowness and position vectors, measured clockwise
    phi = [180 - baz + th for th in theta]

    sampling_rate = st[0].stats.sampling_rate

    shifts = []

    # Shift is dot product. The minus sign is because a positive time delay needs to be corrected by a negative shift in order to stack
    for i in range(0, len(st)):

        shifts.append(-1 * int(round(r[i] * s * np.cos(np.radians(phi[i]))* sampling_rate)))

    return shifts

def linear_stack(st, s, baz):
    '''
    Returns the linear (delay-and-sum) stack for a seismic array, for a beam of given slowness and backazimuth.

    Parameters
    ----------
    st : ObsPy Stream object
        Stream of SAC format seismograms for the seismic array, length K = no. of stations in array
    s  : float
        Magnitude of slowness vector, in s / km
    baz : float
        Backazimuth of slowness vector, (i.e. angle from North back to epicentre of event)

    Returns
    -------
    stack : NumPy array
        The delay-and-sum beam at the given slowness and backazimuth, as a time series.
    '''

    # Check that each channel has the same number of samples, otherwise we can't construct the beam properly
    assert len(set([len(tr) for tr in st])) == 1, "Traces in stream have different lengths, cannot stack."

    nsta = len(st)

    shifts = get_shifts(st, s, baz)

    shifted_st = st.copy()
    for i, tr in enumerate(shifted_st):
        tr.data = np.roll(tr.data, shifts[i])

    stack = np.sum([tr.data for tr in shifted_st], axis=0) / nsta

    return stack

def nth_root_stack(st, s, baz, n):
    '''
    Returns the nth root stack for a seismic array, for a beam of given slowness and backazimuth.

    Parameters
    ----------
    st : ObsPy Stream object
        Stream of SAC format seismograms for the seismic array, length K = no. of stations in array
    s  : float
        Magnitude of slowness vector, in s / km
    baz : float
        Backazimuth of slowness vector, (i.e. angle from North back to epicentre of event)
    n : int
        Order of the nth root process (n=1 just yields the linear vespa)

    Returns
    -------
    stack : NumPy array
        The nth root beam at the given slowness and backazimuth, as a time series.
    '''
    # Check that each channel has the same number of samples, otherwise we can't construct the beam properly
    assert len(set([len(tr) for tr in st])) == 1, "Traces in stream have different lengths, cannot stack."

    nsta = len(st)

    shifts = get_shifts(st, s, baz)

    stack = np.zeros(st[0].data.shape)
    for i, tr in enumerate(st):
        stack += np.roll(pow(abs(tr.data), 1./n) * np.sign(tr.data), shifts[i]) # Shift data in each trace by its offset

    stack /= nsta
    stack = pow(abs(stack), n) * np.sign(stack)

    return stack

