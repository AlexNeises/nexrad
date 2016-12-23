import os
import os.path

from matplotlib.cbook import Bunch, is_string_like, iterable

def get_test_data(fname, as_file_obj = True):
    data_dir = os.environ.get('TEST_DATA_DIR', os.path.join(os.path.dirname(__file__), '..', 'testdata'))

    path = os.path.join(data_dir, fname)

    if as_file_obj:
        return open(path, 'rb')

    return path

__all__ = ('Bunch', 'get_test_data', 'is_string_like', 'iterable')