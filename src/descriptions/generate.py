import fcntl
import json
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import scipy.io as sio
from tqdm import tqdm

from src.data_loader.class_names import class_names
from src.descriptions.prompts import PROMPT_VERSION, classwise_prompt, pairwise_prompt
from src.utils.naming import llm_tag


def required_pair_keys(seen, unseen, symmetric):
    seen = [int(x) for x in seen]
    unseen = [int(x) for x in unseen]
    if symmetric:
        pairs = {(min(a, b), max(a, b)) for a in seen for b in seen if a != b}
    else:
        pairs = {(a, b) for a in seen for b in seen if a != b}
    return pairs | {(a, b) for a in unseen for b in seen}


def _names(cfg):
    return class_names(cfg.dataset, sio.loadmat(Path(cfg.data_root) / cfg.dataset / 'att_splits.mat'))


def _read_rows(path, kind):
    rows = {}
    lines = path.read_bytes().splitlines(keepends=True)
    valid_bytes = 0
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(lines) - 1:
                raise
            warnings.warn(f'Dropping truncated last line in {path}', stacklevel=2)
            break
        key = (row['a'], row['b']) if kind == 'pairwise' else row['a']
        rows[key] = row['text']
        valid_bytes += len(line)
    return rows, valid_bytes


def load_descriptions(cache_dir, dataset, llm_tag, kind):
    if kind not in ('pairwise', 'classwise'):
        raise ValueError(f'Unknown description kind: {kind}')
    path = Path(cache_dir) / dataset / llm_tag / f'{kind}.jsonl'
    if not path.exists():
        raise FileNotFoundError(f'Description cache not found: {path}')
    return _read_rows(path, kind)[0]


def check_manifest(cfg, directory, write=False):
    directory = Path(directory)
    names = _names(cfg)
    manifest_path = directory / 'manifest.json'
    manifest = {'dataset': cfg.dataset, 'prompt_version': PROMPT_VERSION,
                'llm': {key: getattr(cfg.llm, key) for key in ('provider', 'name', 'temperature', 'top_p', 'max_tokens')},
                'symmetric': cfg.descriptions.symmetric, 'classes': names}
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
            raise ValueError(f'Description manifest mismatch: {manifest_path}')
    elif write:
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    else:
        raise FileNotFoundError(f'Description manifest not found: {manifest_path}')
    return manifest


def generate_descriptions(cfg, data, client, kind):
    if kind not in ('pairwise', 'classwise'):
        raise ValueError(f'Unknown description kind: {kind}')
    directory = Path(cfg.descriptions.cache_dir) / cfg.dataset / llm_tag(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a') as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        return _generate_descriptions(cfg, data, client, kind, directory)


def _generate_descriptions(cfg, data, client, kind, directory):
    names = check_manifest(cfg, directory, write=True)['classes']
    path = directory / f'{kind}.jsonl'
    cached, valid_bytes = _read_rows(path, kind) if path.exists() else ({}, 0)
    if path.exists():
        with path.open('r+b') as f:
            f.truncate(valid_bytes)
            if valid_bytes:
                f.seek(-1, 2)
                if f.read(1) != b'\n':
                    f.seek(0, 2)
                    f.write(b'\n')
    required = required_pair_keys(data.seen_classes, data.unseen_classes, cfg.descriptions.symmetric) if kind == 'pairwise' else set(range(len(names)))
    missing = sorted(required - cached.keys())
    if cfg.descriptions.max_items > 0:
        missing = missing[:cfg.descriptions.max_items]
    lock = threading.Lock()
    failed = threading.Event()

    def request(key):
        if failed.is_set():
            return
        if kind == 'pairwise':
            a, b = key
            system, user = pairwise_prompt(cfg.dataset, names[a], names[b])
            row = {'a': a, 'b': b, 'name_a': names[a], 'name_b': names[b]}
        else:
            a = key
            system, user = classwise_prompt(cfg.dataset, names[a])
            row = {'a': a, 'name_a': names[a]}
        try:
            row['text'] = client.complete(system, user)
        except Exception:
            failed.set()
            raise
        with lock:
            with path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                f.flush()
            cached[key] = row['text']

    with ThreadPoolExecutor(max_workers=cfg.llm.max_workers) as pool:
        futures = [pool.submit(request, key) for key in missing]
        for future in tqdm(as_completed(futures), total=len(futures), desc=kind):
            try:
                future.result()
            except Exception:
                pool.shutdown(wait=True, cancel_futures=True)
                raise
    print(f'{kind}: {len(cached)} descriptions')
    return cached
