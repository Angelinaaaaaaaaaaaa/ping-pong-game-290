"""Streaming ZIP/directory access and deterministic match-balanced sampling."""
import io
import hashlib
from pathlib import Path
import zipfile
import numpy as np


class RallyStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        archive = self.root if self.root.suffix == '.zip' else self.root / 'release_data.zip'
        extracted = self.root / 'release_data'
        if extracted.is_dir():
            self.root = extracted
        self.zip = None
        if self.root.is_dir() and any(self.root.glob('match*')):
            self.names = sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*') if p.is_file())
        elif archive.is_file():
            self.zip = zipfile.ZipFile(archive)
            self.names = sorted(self.zip.namelist())
        else:
            raise ValueError('Expected release_data.zip or an extracted release_data/match*/ tree')
        self.name_set = set(self.names)
        self.recons = [n for n in self.names if n.endswith('_recon.npy') and not n.startswith('__MACOSX/')]

    def read(self, name):
        return self.zip.read(name) if self.zip else (self.root / name).read_bytes()

    def array(self, name):
        return np.load(io.BytesIO(self.read(name)), allow_pickle=False)

    def sample(self, count, seed):
        groups = {}
        for name in self.recons:
            groups.setdefault(Path(name).parent.parent.name, []).append(name)
        rng = np.random.default_rng(seed)
        matches = rng.permutation(sorted(groups))
        groups = {m: list(rng.permutation(groups[m])) for m in matches}
        result = []
        while len(result) < min(count, len(self.recons)):
            for match in matches:
                if groups[match] and len(result) < count:
                    result.append(str(groups[match].pop()))
        return result

    def close(self):
        if self.zip:
            self.zip.close()


def digest(data):
    return hashlib.sha256(data).hexdigest()
