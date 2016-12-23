from __future__ import division

import numpy as np
import pint
import pint.unit

UndefinedUnitError = pint.UndefinedUnitError

units = pint.UnitRegistry(autoconvert_offset_to_baseunit = True)

units.define(pint.unit.UnitDefinition('percent', '%', (), pint.converters.ScaleConverter(0.01)))

def concatenate(arrs, axis = 0):
    dest = 'dimensionless'
    for a in arrs:
        if hasattr(a, 'units'):
            dest = a.units
            break

    data = []
    for a in arrs:
        if hasattr(a, 'to'):
            a = a.to(dest).magnitude
        data.append(np.atleast_1d(a))

    return units.Quantity(np.concatenate(data, axis = axis), dest)

def atleast_1d(*arrs):
    mags = [a.magnitude for a in arrs]
    orig_units = [a.units for a in arrs]
    ret = np.atleast_1d(*mags)
    if len(mags) == 1:
        return units.Quantity(ret, orig_units[0])
    return [units.Quantity(m, u) for m, u in zip(ret, orig_units)]

def atleast_2d(*arrs):
    mags = [a.magnitude for a in arrs]
    orig_units = [a.units for a in arrs]
    ret = np.atleast_2d(*mags)
    if len(mags) == 1:
        return units.Quantity(ret, orig_units[0])
    return [units.Quantity(m, u) for m, u in zip(ret, orig_units)]

def masked_array(data, data_units = None, **kwargs):
    if data_units is None:
        data_units = data.units
    return units.Quantity(np.ma.masked_array(data, **kwargs), data_units)

del pint