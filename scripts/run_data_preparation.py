import sys
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scipy.io as sio
from src.data_loader.class_names import class_names

DATASETS = ('CUB', 'AWA2', 'SUN')
FILES = ('att_splits.mat', 'res101.mat', 'res101_finetuned.mat')
FILE_ID = '1KxFC6T_kGKCNx1JyX2FOaSimA0DOcU_I'


def prepare_data(data_root='data'):
    root = Path(data_root)
    missing = [(dataset, name) for dataset in DATASETS for name in FILES
               if not (root / dataset / name).exists()]
    if missing:
        archive = root / '_downloads' / 'sdgzsl' / 'sdgzsl_datasets.zip'
        if not archive.exists():
            import gdown
            archive.parent.mkdir(parents=True, exist_ok=True)
            gdown.download(id=FILE_ID, output=str(archive), quiet=False)
        with ZipFile(archive) as zf:
            for dataset, name in missing:
                target = root / dataset / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(f'SDGZSL_data/{dataset}/{name}') as source, target.open('wb') as dest:
                    import shutil
                    shutil.copyfileobj(source, dest)
    write_class_names(root)


def write_class_names(data_root='data'):
    root = Path(data_root)
    for dataset in DATASETS:
        split = root / dataset / 'att_splits.mat'
        if split.exists():
            names = class_names(dataset, sio.loadmat(split))
            target = root / 'class_names' / f'{dataset}.txt'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('\n'.join(names) + '\n', encoding='utf-8')


if __name__ == '__main__':
    prepare_data()
