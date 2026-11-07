"""Build a reproducible Colab archive from an explicit source/data allowlist."""
import gzip
import hashlib
import io
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def bundle_files():
    files = ['README.md', 'configs/baseline.json', 'requirements-train.txt',
             'requirements-inference.txt', 'docs/colab_runbook.md',
             'docs/baseline_design.md', 'notebooks/lifemate_colab_train.ipynb']
    for folder in ('src/lifemate', 'training', 'scripts'):
        files.extend(str(path.relative_to(ROOT)) for path in (ROOT / folder).glob('*.py'))
    return {name: (ROOT / name).read_bytes() for name in sorted(set(files))}


def checksums(files):
    return ''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n'
                   for name, data in sorted(files.items())).encode()


def build():
    files = bundle_files()
    digest = hashlib.sha256(checksums(files)).hexdigest()[:12]
    files['COLAB_BUNDLE_SHA256SUMS'] = checksums(files)
    output = ROOT / 'artifacts/colab'
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f'lifemate-colab-bundle-v3-{digest}.tar.gz'
    with archive.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w') as tar:
            for name, data in sorted(files.items()):
                info = tarfile.TarInfo(f'lifemate/{name}')
                info.size, info.mode = len(data), 0o644
                tar.addfile(info, io.BytesIO(data))
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(archive.suffix + '.sha256').write_text(f'{checksum}  {archive.name}\n')
    shutil.copyfile(ROOT/'notebooks/lifemate_colab_train.ipynb', output/'lifemate_colab_train_v3.ipynb')
    print(f'{archive}\nSHA256: {checksum}\nBytes: {archive.stat().st_size}')
    return archive


if __name__ == '__main__':
    build()
