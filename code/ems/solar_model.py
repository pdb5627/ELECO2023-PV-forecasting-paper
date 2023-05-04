import pandas as pd
import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain


def clearsky(location, df_in, pu=True):
    """ Calculate clearsky PV output using the PVLib model.
    location: dict with location information. Should have members
        'lat', 'lon', 'name', 'elevation', 'tilt', 'azimuth', 'nominal_max_output'.
    df_in: dataframe with DatetimeIndex for times of desired forecast.
        Optionally may have a column 'temp' with ambient temperatures.
    """
    l = location # Shorter name for concise use in this function
    times = df_in.index


    # get the module and inverter specifications from SAM
    # Default module here is 220W nominal output
    use_sandia_model = True
    if use_sandia_model:
        mods = pvlib.pvsystem.retrieve_sam('sandiamod')
        module = mods['Canadian_Solar_CS5P_220M___2009_']
        aoi_model = None # Infer from data
    else:
        mods_cec = pvlib.pvsystem.retrieve_sam('cecmod')
        module = mods_cec['Canadian_Solar_Inc__CS5P_220M']
        aoi_model = 'no_loss' # cec doesn't include parameters for loss models
    invs = pvlib.pvsystem.retrieve_sam('cecinverter')
    inverter = invs['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
    temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

    # specify constant ambient air temp and wind for simplicity
    temp_air = 20
    wind_speed = 0

    system = PVSystem(module_parameters=module,
                      inverter_parameters=inverter,
                      temperature_model_parameters=temperature_model_parameters,
                      surface_tilt=location['tilt'],
                      surface_azimuth=location['azimuth']
    )

    # TODO: FIXME. Decide in what structure to return the data.
    df = pd.DataFrame(index=times, columns=['modeled_output',
                                            'effective_irradiance'])

    location = Location(l['lat'], l['lon'], name=l.get('name', ''), altitude=l['elevation'])
    weather = location.get_clearsky(times)
    if False and 'Temp' in df_in.columns:
        weather['temp_air'] = df_in['Temp']
    mc = ModelChain(system, location)
    mc.run_model(weather)

    # Scale to kW at the actual project size
    df['modeled_output'] = (1 if pu else l['nominal_max_output'])*mc.results.ac/220
    df['effective_irradiance'] = mc.results.effective_irradiance
    df['modeled_ghi'] = mc.results.weather['ghi']

    return df
