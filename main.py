import numpy as np
import matplotlib.pyplot as plt
from numpy import ma

from decode import L3D
import colors

fig, axes = plt.subplots(1, 2, figsize = (15, 8))
for v, color, ax in zip(('N0Q', 'N0U'), ('NWSReflectivity', 'NWSVelocity'), axes):
	f = L3D('examples/radar_data/KOUN_SDUS54_%sTLX_201305202016' % v)
	
	datadict = f.sym_block[0][0]
	data = ma.array(datadict['data'])
	data[data == 0] = ma.masked

	az = np.array(datadict['start_az'] + [datadict['end_az'][-1]])
	rng = np.linspace(0, f.max_range, data.shape[-1] + 1)

	xlocs = rng * np.sin(np.deg2rad(az[:, np.newaxis]))
	ylocs = rng * np.cos(np.deg2rad(az[:, np.newaxis]))

	norm, cmap = colors.registry.get_with_steps(color, 16, 16)
	ax.pcolormesh(xlocs, ylocs, data, norm = norm, cmap = cmap)
	ax.set_aspect('equal', 'datalim')
	ax.set_xlim(-40, 20)
	ax.set_ylim(-30, 30)

plt.show()