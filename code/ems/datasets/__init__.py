import sys, os
from datetime import timedelta
""" Package to read in the various datasets that I have collected and return a 
consistent dataframe. The dataframe will have the following index and columns:
index: DateTimeIndex is a naive timestamp at the local time.
'P_out': PV output power, normalized to nominal output, if available, or maximum 
         output, if the nominal output is not available.
Additional columns may be present, but are not guaranteed. (Column names may be 
standardized in the future.)

The data will be resampled to an hourly basis using a mean.
TODO: Would it make sense to shift the time series so that times are based on 
solar position (e.g. 12:00 = solar noon??).  GMT time +/- long*12/180

"""
#from .dblock import *
#from .solcast import *
from .abb_inverter_logger import *
from .clearsky_model import *
from .solcast_weather import *
from .meteogram_forecast import *


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]


    return 0


if __name__ == '__main__':
    sys.exit(main())
