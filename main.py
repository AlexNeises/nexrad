from nexrad import L3D
import colors
import numpy as np
import matplotlib.pyplot as plt
from numpy import ma

def scaled_elem(index, scale):
	def inner(seq):
		return seq[index] * scale
	return inner

radar = L3D('KOUN_SDUS54_N0QTLX_201305202016')

data = ma.array(radar.get_data()) * scaled_elem(2, 0.1)
data[data==0] = ma.masked
az = np.array(radar.get_start_azimuth() + radar.get_end_azimuth()[-1])
rng = np.linspace(0, 460.0, data.shape[-1] + 1)

xlocs = rng * np.sin(np.deg2rad(az[:, np.newaxis])) * -1
ylocs = rng * np.cos(np.deg2rad(az[:, np.newaxis])) * -1

fig, axes = plt.subplots(1, 1, figsize = (8, 8))

norm, cmap = colors.registry.get_with_steps('NWSReflectivityExpanded', 16, 16)
axes.pcolormesh(xlocs, ylocs, data, norm = norm, cmap = cmap)
axes.set_axis_bgcolor('black')
axes.set_aspect('equal', 'datalim')
axes.set_xlim(-25, 25)
axes.set_ylim(-25, 25)

plt.show()