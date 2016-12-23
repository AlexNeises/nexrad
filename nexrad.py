import bz2
import struct
from datetime import datetime, timedelta
import struct
import numpy as np

class L3D(object):
    def __init__(self, filename):
        fhandle = open(filename, 'rb')
        buf = fhandle.read()
        fhandle.close()

        self.text_header = buf[:30]
        bpos = 30

        self.msg_header = unpack_buffer(buf, bpos, MESSAGE_HEADER)
        if self.msg_header['code'] not in SUPPORTED_PRODUCTS:
            code = self.msg_header['code']
            raise NotImplementedError('Level III product with code %i is not supported' % (code))

        bpos += 18

        self.prod_descr = unpack_buffer(buf, bpos, PRODUCT_DESCRIPTION)
        bpos += 102

        if buf[bpos:bpos + 2] == 'BZ':
            buf2 = bz2.decompress(buf[bpos:])
        else:
            buf2 = buf[bpos:]

        self.read_symbology_block(buf2)

    def read_symbology_block(self, buf2):
        self.symbology_header = unpack_buffer(buf2, 0, SYMBOLOGY_HEADER)

        packet_code = struct.unpack('>h', buf2[16:18])[0]
        assert packet_code in SUPPORTED_PACKET_CODES
        self.packet_header = unpack_buffer(buf2, 16, RADIAL_PACKET_HEADER)
        self.radial_headers = []
        nbins = self.packet_header['nbins']
        nradials = self.packet_header['nradials']
        nbytes = unpack_buffer(buf2, 30, RADIAL_HEADER)['nbytes']
        if packet_code == 16 and nbytes != nbins:
            nbins = nbytes
        self.raw_data = np.empty((nradials, nbins), dtype = 'uint8')
        pos = 30

        for radial in self.raw_data:
            radial_header = unpack_buffer(buf2, pos, RADIAL_HEADER)
            pos += 6
            if packet_code == 16:
                radial[:] = np.fromstring(buf2[pos:pos + nbins], '>u1')
                pos += radial_header['nbytes']
            else:
                assert packet_code == AF1F
                rle_size = radial_header['nbytes'] * 2
                rle = np.fromstring(buf2[pos:pos + rle_size], dtype = '>u1')
                colors = np.bitwise_and(rle, 0b00001111)
                runs = np.bitwise_and(rle, 0b11110000) / 16
                radial[:] = np.repeat(colors, runs)
                pos += rle_size
            self.radial_headers.append(radial_header)

    def get_location(self):
        latitude = self.prod_descr['latitude'] * 0.001
        longitude = self.prod_descr['longitude'] * 0.001
        height = self.prod_descr['height']
        return latitude, longitude, height

    def get_start_azimuth(self):
        azimuths = [d['angle_start'] for d in self.radial_headers]
        return np.array(azimuths, dtype = 'float32') * 0.1

    def get_end_azimuth(self):
        startazimuths = [d['angle_start'] for d in self.radial_headers]
        startazimuths = np.array(startazimuths, dtype = 'float32') * 0.1
        endazimuths = [d['angle_delta'] for d in self.radial_headers]
        return startazimuths + np.array(endazimuths, dtype = 'float32') * 0.1

    def get_range(self):
        nbins = self.raw_data.shape[1]
        first_bin = self.packet_header['first_bin']
        range_scale = (self.packet_header['range_scale'] * PRODUCT_RANGE_RESOLUTION[self.msg_header['code']])
        return np.arange(nbins, dtype = 'float32') * range_scale + first_bin

    def get_elevation(self):
        hw30 = self.prod_descr['halfwords_30']
        elevation = struct.unpack('>h', hw30)[0] * 0.1
        return elevation

    def get_volume_start_datetime(self):
        return datetime_from_mdatetime(self.prod_descr['vol_scan_date'], self.prod_descr['vol_scan_time'])

    def get_data(self):
        msg_code = self.msg_header['code']
        threshold_data = self.prod_descr['threshold_data']

        if msg_code in data_levels_816:
            mdata = self.get_816_data_levels()
        elif msg_code in [134]:
            mdata = self.get_data_msg_134()
        elif msg_code in [94, 99, 182, 186]:
            hw31, hw32 = np.fromstring(threshold_data[:4], '>i2')
            data = (self.raw_data - 2) * (hw32 / 10.) + hw31 / 10.
            mdata = np.ma.array(data, mask = self.raw_data < 2)
        elif msg_code in [32]:
            hw31, hw32 = np.fromstring(threshold_data[:4], '>i2')
            data = (self.raw_data) * (hw32 / 10.) + hw31 / 10.
            mdata = np.ma.array(data, mask = self.raw_data < 2)
        elif msg_code in [138]:
            hw31, hw32 = np.fromstring(threshold_data[:4], '>i2')
            data = self.raw_data * (hw32 / 100.) + hw31 / 100.
            mdata = np.ma.array(data)
        elif msg_code in [159, 161, 163]:
            scale, offset = np.fromstring(threshold_data[:8], '>f4')
            data = (self.raw_data - offset) / (scale)
            mdata = np.ma.array(data, mask = self.raw_data < 2)
        elif msg_code in [170, 172, 173, 174, 175]:
            scale, offset = np.fromstring(threshold_data[:8], '>f4')
            data = (self.raw_data - offset) / (scale) * 0.01
            mdata = np.ma.array(data, mask = self.raw_data < 1)
        elif msg_code in [165, 177]:
            mdata = np.ma.masked_equal(self.raw_data, 0)
        elif msg_code in [135]:
            mdata = np.ma.array(self.raw_data - 2, mask = self.raw_data <= 1)
            mdata[self.raw_data >= 128] -= 128
        else:
            assert msg_code in [34]
            mdata = np.ma.array(self.raw_data.copy())

        return mdata.astype('float32')

    def get_816_data_levels(self):
        thresh = np.fromstring(self.prod_descr['threshold_data'], '>B')
        flags = thresh[::2]
        values = thresh[1::2]

        sign = np.choose(np.bitwise_and(flags, 0x01), [1, -1])
        bad = np.bitwise_and(flags, 0x80) == 128
        scale = 1.
        if flags[0] & 2**5:
            scale = 1 / 20.
        if flags[0] & 2**4:
            scale = 1 / 10.

        data_levels = values * sign * scale
        data_levels[bad] = -999

        data = np.choose(self.raw_data, data_levels)
        mdata = np.ma.masked_equal(data, -999)
        return mdata

    def get_data_msg_134(self):
        hw31, hw32, hw33, hw34, hw35 = np.fromstring(self.prod_descr['threshold_data'][:10], '>i2')
        linear_scale = int16_to_float16(hw31)
        linear_offset = int16_to_float16(hw32)
        log_start = hw33
        log_scale = int16_to_float16(hw34)
        log_offset = int16_to_float16(hw35)

        data = np.zeros(self.raw_data.shape, dtype = np.float32)
        lin = self.raw_data < log_start
        data[lin] = ((self.raw_data[lin] - linear_offset) / (linear_scale))

        log = self.raw_data >= log_start
        data[log] = np.exp((self.raw_data[log] - log_offset) / (log_scale))
        mdata = np.ma.masked_array(data, mask = self.raw_data < 2)
        return mdata

def datetime_from_mdatetime(mdate, mtime):
    epoch = datetime.utcfromtimestamp(0)
    return epoch + timedelta(days = mdate - 1, seconds = mtime)

def structure_size(structure):
    return struct.calcsize('>' + ''.join([i[1] for i in structure]))

def unpack_buffer(buf, pos, structure):
    size = structure_size(structure)
    return unpack_structure(buf[pos:pos + size], structure)

def unpack_structure(string, structure):
    fmt = '>' + ''.join([i[1] for i in structure])
    lst = struct.unpack(fmt, string)
    return dict(zip([i[0] for i in structure], lst))

def nexrad_level3_message_code(filename):
    fhl = open(filename, 'r')
    buf = fhl.read(48)
    rhl.close()
    msg_header = unpack_buffer(buf, 30, MESSAGE_HEADER)
    return msg_header['code']

def int16_to_float16(val):
    sign = (val & 0b1000000000000000) / 0b1000000000000000
    exponent = (val & 0b0111110000000000) / 0b0000010000000000
    fraction = (val & 0b0000001111111111)
    if exponent == 0:
        return (-1)**sign * 2 * (0 + (fraction / 2**10.))
    else:
        return (-1)**sign * 2**(exponent - 16) * (1 + fraction / 2**10.)

data_levels_816 = [19, 20, 25, 27, 28, 30, 56, 78, 79, 80, 169, 171, 181]

PRODUCT_RANGE_RESOLUTION = {
    19:  1.,
    20:  2.,
    25:  0.25,
    27:  1.,
    28:  0.25,
    30:  1,
    32:  1,
    34:  1.,
    56:  1.,
    78:  1.,
    79:  1.,
    80:  1.,
    94:  1.,
    99:  0.25,
    134: 1000.,
    135: 1000.,
    138: 1.,
    159: 0.25,
    161: 0.25,
    163: 0.25,
    165: 0.25,
    169: 1.,
    170: 1.,
    171: 1.,
    172: 1.,
    173: 1.,
    174: 1.,
    175: 1.,
    177: 0.25,
    181: 150.,
    182: 150.,
    186: 300.
}

# Figure E-1
BYTE  = 'B'
INT2  = 'h'
INT4  = 'i'
UINT4 = 'I'
REAL4 = 'f'

# Figure 3-3
MESSAGE_HEADER = (
    ('code',    INT2),
    ('date',    INT2),
    ('time',    INT4),
    ('length',  INT4),
    ('source',  INT2),
    ('dest',    INT2),
    ('nblocks', INT2)
)

# Figure 3-6
PRODUCT_DESCRIPTION = (
    ('divider',          INT2),
    ('latitude',         INT4),
    ('longitude',        INT4),
    ('height',           INT2),
    ('product_code',     INT2),
    ('operational_mode', INT2),
    ('vcp',              INT2),
    ('sequence_num',     INT2),
    ('vol_scan_num',     INT2),
    ('vol_scan_date',    INT2),
    ('vol_scan_time',    INT4),
    ('product_date',     INT2),
    ('product_time',     INT4),
    ('halfwords_27_28',  '4s'),
    ('elevation_num',    INT2),
    ('halfwords_30',     '2s'),
    ('threshold_data',  '32s'),
    ('halfwords_47_53', '14s'),
    ('version',          BYTE),
    ('spot_blank',       BYTE),
    ('offset_symbology', INT4),
    ('offset_graphic',   INT4),
    ('offset_tabular',   INT4)
)

# Figure 3-6
SYMBOLOGY_HEADER = (
    ('divider',       INT2),
    ('id',            INT2),
    ('block_length',  INT4),
    ('layers',        INT2),
    ('layer_divider', INT2),
    ('layer_length',  INT4)
)

# Figure 3-10
AF1F = -20705
SUPPORTED_PACKET_CODES = [16, AF1F]
RADIAL_PACKET_HEADER = (
    ('packet_code',    INT2),
    ('first_bin',      INT2),
    ('nbins',          INT2),
    ('i_sweep_center', INT2),
    ('j_sweep_center', INT2),
    ('range_scale',    INT2),
    ('nradials',       INT2)
)

RADIAL_HEADER = (
    ('nbytes',      INT2),
    ('angle_start', INT2),
    ('angle_delta', INT2)
)

# Pages 3-15 to 3-22
SUPPORTED_PRODUCTS = [
    19,     # Base Reflectivity
    20,     # Base Reflectivity
    94
]