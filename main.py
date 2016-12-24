import numpy as np
import matplotlib.pyplot as plt
from numpy import ma

from decode import (L3D, L2D)
import colors

# NMD - 141: Mesocyclone Detection
# DVL - 134: High Resolution VIL
# N0Q - 94: Base Reflectivity Data Array
# N0R - 19: Base Reflectivity
# N0S - 56: Storm Relative Mean Radial Velocity
# N0U - 99: Base Velocity Data Array
# N0V - 27: Base Velocity
# NVL - 57: Vertically Integrated Liquid
# NML - 66: Layer Composite Reflectivity (Layer 2 Max)
# NSS - 62: Storm Structure
# NTV - 61: Tornado Vortex Signature
# N0Z - 20: Base Reflectivity
# N0C - 161: Digital Correlation Coefficient
# N0K - 163: Digital Specific Differential Phase
# NAH - 165: Digital Hydrometeor Classification

v = 'N0Q'
color = 'NWSReflectivityExtended'

fig, ax = plt.subplots(1, 2, figsize = (15, 8))
f = L3D('examples/radar_data/KOUN_SDUS54_%sTLX_201305202016' % v)

datadict = f.sym_block[0][0]
data = ma.array(datadict['data'])
data[data == 0] = ma.masked

az = np.array(datadict['start_az'] + [datadict['end_az'][-1]])
rng = np.linspace(0, f.max_range, data.shape[-1] + 1)

xlocs = rng * np.sin(np.deg2rad(az[:, np.newaxis]))
ylocs = rng * np.cos(np.deg2rad(az[:, np.newaxis]))

norm, cmap = colors.registry.get_with_steps(color, 16, 16)
ax[0].pcolormesh(xlocs, ylocs, data, norm = norm, cmap = cmap)
ax[0].set_aspect('equal', 'datalim')
ax[0].set_title('Level III Ref')
ax[0].set_axis_bgcolor('black')
ax[0].set_xlim(-40, 20)
ax[0].set_ylim(-30, 30)

# f = L2D('examples/radar_data/KTLX20130520_201643_V06.gz')
# f.sweeps[0][0]

# # sweep = 0
# sweep = 1
# az = np.array([ray[0].az_angle for ray in f.sweeps[sweep]])
# ref_hdr = f.sweeps[sweep][0][4][b'REF'][0]
# ref_range = np.arange(ref_hdr.num_gates) * ref_hdr.gate_width + ref_hdr.first_gate
# ref = np.array([ray[4][b'REF'][1] for ray in f.sweeps[sweep]])

# data = ma.array(ref)
# data[np.isnan(data)] = ma.masked

# print data.shape

# xlocs = ref_range * np.sin(np.deg2rad(az[:, np.newaxis]))
# ylocs = ref_range * np.cos(np.deg2rad(az[:, np.newaxis]))

# cmap = colors.registry.get_colortable('NWSReflectivity')
# ax[0].pcolormesh(xlocs, ylocs, data, cmap = cmap)
# ax[0].set_aspect('equal', 'datalim')
# ax[0].set_title('Level II Ref')
# ax[0].set_axis_bgcolor('black')
# ax[0].set_xlim(-40, 20)
# ax[0].set_ylim(-30, 30)

f = L2D('examples/radar_data/KTLX20130520_201643_V06.gz')
f.sweeps[0][0]

# sweep = 0
sweep = 1
az = np.array([ray[0].az_angle for ray in f.sweeps[sweep]])
ref_hdr = f.sweeps[sweep][0][4][b'REF'][0]
ref_range = np.arange(ref_hdr.num_gates) * ref_hdr.gate_width + ref_hdr.first_gate
ref = np.array([ray[4][b'REF'][1] for ray in f.sweeps[sweep]])

data = ma.array(ref)
data[np.isnan(data)] = ma.masked

print data.shape

xlocs = ref_range * np.sin(np.deg2rad(az[:, np.newaxis]))
ylocs = ref_range * np.cos(np.deg2rad(az[:, np.newaxis]))

cmap = colors.registry.get_colortable('NWSReflectivityExtended')
ax[1].pcolormesh(xlocs, ylocs, data, cmap = cmap)
ax[1].set_aspect('equal', 'datalim')
ax[1].set_title('Level II Ref')
ax[1].set_axis_bgcolor('black')
ax[1].set_xlim(-40, 20)
ax[1].set_ylim(-30, 30)

plt.show()