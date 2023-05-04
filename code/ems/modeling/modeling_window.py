import dataclasses
import pandas as pd
from typing import Optional, Union
from itertools import groupby


@dataclasses.dataclass(frozen=True)
class ModelingWindow:
    start: pd.Timestamp
    tz: str
    day_start_offset: pd.Timedelta
    end: Optional[pd.Timestamp] = None
    whole_days: Optional[int] = None
    delta_t: pd.Timedelta = pd.Timedelta('1h')  # A delta_t different from 1h is going to impact a lot of code. Beware.

    def __post_init__(self):
        if self.whole_days is None and self.end is None:
            raise ValueError('Neither end nor whole days is set.')
        elif self.whole_days is not None and self.end is None:
            # Cannot directly assign to self.end due to dataclass being frozen
            object.__setattr__(self, 'end', self.determine_end_from_whole_days())

    @property
    def D_original(self):
        """"
        Calculates which day to assign each model time index to.
        Assumes that the ending time is at the end of a whole day (i.e. at day_start_offset in determine_model_window).
        Assumes that the index will be created closed on left and open on right.
        Assumes that delta_t evenly divides a day.
        """
        num_points = (self.end - self.start) // self.delta_t
        points_per_day = pd.to_timedelta('1d') // self.delta_t
        D = tuple(tuple(num_points - t for t in g) for k, g in groupby(range(num_points, 0, -1),
                                                                       lambda t: (t - 1) // points_per_day))
        return D

    @property
    def index(self):
        return pd.date_range(self.start, self.end, freq=self.delta_t, inclusive='left')

    @property
    def D(self):
        """"
        Calculates which day to assign each model time index to.
        Assumes that the ending time is at the end of a whole day (i.e. at day_start_offset in determine_model_window).
        Assumes that the index will be created closed on left and open on right.
        Assumes that delta_t evenly divides a day.
        """
        days = self.Vuse_day(self.index)
        D = tuple(tuple(v[0] for v in g) for k, g in groupby(enumerate(days), lambda v: v[1]))
        return D

    @property
    def num_points(self):
        return (self.end - self.start) // self.delta_t

    def determine_end_from_whole_days(self):
        """"
        Calculates the end datetimes for a model window that starts at start and covers the rest of that day
        and the following whole_days number of whole days.
        """
        if self.whole_days is None:
            return None
        start_offset = self.start_localized - self.day_start_offset
        end_offset = start_offset.floor('1d') + (self.whole_days + 1) * pd.to_timedelta('1d')
        end = end_offset + self.day_start_offset
        # start and end are both stored WITHOUT any time zone information, as UTC, so strip the timezone info
        end = end.tz_convert('UTC').tz_convert(None)
        return end

    @property
    def start_localized(self):
        return self.start.tz_localize('UTC').tz_convert(self.tz)

    @property
    def end_localized(self):
        return self.end.tz_localize('UTC').tz_convert(self.tz)

    def Vuse_day(self, dt: Union[pd.Timestamp, pd.DatetimeIndex]):
        if dt.tz is None:
            dt = dt.tz_localize('UTC').tz_convert(self.tz)
        return (dt - self.day_start_offset).floor('1d').tz_localize(None)

    @property
    def Vuse_start(self):
        return self.Vuse_day(self.start)

    @property
    def Vuse_end(self):
        Δ = self.delta_t / 10
        return self.Vuse_day(self.end - Δ)
