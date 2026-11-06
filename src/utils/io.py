import os
import tempfile
from pathlib import Path

import torch


def atomic_torch_save(obj, path):
    path = Path(path)
    fd, temp = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    os.close(fd)
    try:
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(temp, 0o666 & ~umask)
        torch.save(obj, temp)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
