from __future__ import division

import ast
import glob
import logging
import os.path
import posixpath

import matplotlib.colors as mcolors

from pkg_resources import resource_listdir, resource_stream

TABLE_EXT = '.tbl'

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.WARNING)

def _parse(s):
	if hasattr(s, 'decode'):
		s = s.decode('ascii')

	if not s.startswith('#'):
		return ast.literal_eval(s)

	return None

def read_colortable(fobj):
	ret = list()
	try:
		for line in fobj:
			literal = _parse(line)
			if literal:
				ret.append(mcolors.colorConverter.to_rgb(literal))
		return ret
	except (SyntaxError, ValueError):
		raise RuntimeError('Malformed colortable.')

def convert_gempak_table(infile, outfile):
	for line in infile:
		if not line.startswith('!') and line.strip():
			r, g, b = map(int, line.split())
			outfile.write('({0:f}, {1:f}, {2:f})\n'.format(r / 255, g / 255, b / 255))

class ColortableRegistry(dict):
	def scan_dir(self, path):
		for fname in glob.glob(os.path.join(path, '*' + TABLE_EXT)):
			if os.path.isfile(fname):
				with open(fname, 'r') as fobj:
					try:
						self.add_colortable(fobj, os.path.splitext(os.path.basename(fname))[0])
						log.debug('Added colortable from file: %s', fname)
					except RuntimeError:
						log.info('Skipping unparsable file: %s', fname)

	def add_colortable(self, fobj, name):
		self[name] = read_colortable(fobj)

	def get_with_steps(self, name, start, step):
		from numpy import arange
		num_steps = len(self[name]) + 1
		boundaries = arange(start, start + step * num_steps, step)
		return self.get_with_boundaries(name, boundaries)

	def get_with_boundaries(self, name, boundaries):
		cmap = self.get_colortable(name)
		return mcolors.BoundaryNorm(boundaries, cmap.N), cmap

	def get_colortable(self, name):
		return mcolors.ListedColormap(self[name], name = name)

registry = ColortableRegistry()
registry.scan_dir(os.path.curdir + '/resources/colortables/')