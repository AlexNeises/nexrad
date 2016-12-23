__all__ = ('Exporter',)

class Exporter(object):
    def __init__(self, globls):
        self.globls = globls
        self.exports = globls.setdefault('__all__', [])

    def export(self, defn):
        self.exports.append(defn.__name__)
        return defn

    def __enter__(self):
        self.start_vars = set(self.globls)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exports.extend(set(self.globls) - self.start_vars)
        del self.start_vars