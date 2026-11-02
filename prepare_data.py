"""Build small, auditable household stores from the Ausgrid 2012-13 archive."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile
from datetime import datetime, timedelta
import numpy as np

ROOT = Path(__file__).resolve().parent
URL = 'https://raw.githubusercontent.com/rockey-abhishek/Ausgrid-Dataset/main/Data.zip'
ARCHIVE_SHA256 = '7ec04dbe64700d3e69a6d5b0b501ef1a9f8bd503cfc4a1050630f99c2dbcf7b8'
MEMBER = 'RawData/2012-2013 Solar home electricity data v2.csv'


def build(archive, output, clients=10, days=365):
    if not 2 <= clients <= 300 or not 60 <= days <= 365:
        raise ValueError('Use 2..300 homes and 60..365 days.')
    archive, output = Path(archive), Path(output)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError('Archive checksum differs from the verified mirror; inspect before use.')
    start = datetime(2012, 7, 1)
    end = start + timedelta(days=days)
    records = {}
    with zipfile.ZipFile(archive) as z:
        with io.TextIOWrapper(z.open(MEMBER), encoding='utf-8-sig', newline=None) as f:
            # One explanatory line precedes the CSV header; ignore empty lines.
            next(f)
            reader = csv.DictReader(line for line in f if line.strip())
            columns = reader.fieldnames[5:53]
            for row in reader:
                if row['Consumption Category'] != 'GC':
                    continue
                if row.get('Row Quality', '').strip():
                    continue  # Exclude estimated/substituted (NA) rows.
                date = datetime.strptime(row['date'], '%d/%m/%Y')
                if not start <= date < end:
                    continue
                home = int(row['Customer'])
                a = np.array([float(row[x]) if row[x].strip() else np.nan for x in columns])
                key = (date - start).days
                bucket = records.setdefault(home, {})
                if key in bucket:
                    raise ValueError(f'Duplicate GC row: home {home}, day {key}')
                bucket[key] = a
    chosen = []
    for home in sorted(records):
        r = records[home]
        if len(r) != days or set(r) != set(range(days)):
            continue
        a = np.stack([r[d] for d in range(days)])
        if np.isfinite(a).all() and (a >= 0).all():
            chosen.append((home, a.reshape(days, 24, 2).sum(axis=2).astype(np.float32)))
        if len(chosen) == clients:
            break
    if len(chosen) != clients:
        raise ValueError(f'Only {len(chosen)} complete homes available.')
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for home, a in chosen:
        path = output / f'home_{home:03d}.npz'
        np.savez_compressed(path, kwh=a.reshape(-1), start=np.array(start.isoformat()))
        entries.append({'id': home, 'file': path.name, 'hours': int(a.size),
                        'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {
        'dataset': 'Ausgrid Solar Home Electricity Data 2012-2013', 'kind': 'real_public_data',
        'source': 'https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data',
        'mirror': URL, 'archive_sha256': digest, 'archive_bytes': archive.stat().st_size,
        'csv_member': MEMBER, 'target': 'GC only: general consumption; excludes CL and GG',
        'unit': 'kWh per hour', 'start': start.isoformat(), 'end_exclusive': end.isoformat(),
        'time_convention': 'Source local-clock 48-slot days; adjacent half-hour ENERGY values summed. Hour labels are interval starts; no UTC/DST conversion.',
        'selection': 'First numerically ordered complete, finite, nonnegative GC households with blank (actual) Row Quality; not random or representative.',
        'missing': 'Incomplete households excluded; no interpolation or future-data imputation.',
        'clients': entries, 'prepared_bytes': sum(e['bytes'] for e in entries),
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path)
    p.add_argument('--download', action='store_true')
    p.add_argument('--clients', type=int, default=10)
    p.add_argument('--days', type=int, default=365)
    p.add_argument('--output', type=Path, default=ROOT / 'data')
    args = p.parse_args()
    archive = args.archive or ROOT / 'cache' / 'ausgrid.zip'
    if args.download and not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as r:
            payload = r.read(25_000_001)
        if len(payload) > 25_000_000 or hashlib.sha256(payload).hexdigest() != ARCHIVE_SHA256:
            raise ValueError('Download failed size/checksum validation.')
        archive.write_bytes(payload)
    if not archive.exists():
        p.error('Use --download or --archive PATH; prepared data is already bundled for the default demo.')
    build(archive, args.output, args.clients, args.days)
