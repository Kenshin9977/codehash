#!/usr/bin/env python3
"""Read Wraith name index files, the format the community publishes its hash databases in.

    python tools/read_wni.py ghhashes.zip --out names.txt
    python tools/read_wni.py fnv1a_ximages.wni --csv names.csv

A .wni is a magic, a count, and one LZ4 block holding `uint64 key` then a null-terminated name,
repeated. The key is masked to 60 bits, which is the same truncation the games apply.

LZ4 block decoding is written out here rather than pulled in as a dependency. It is forty lines,
the format has not changed in a decade, and a hash-cracking toolchain that needs a pip install to
read its own inputs is worse off than one that does not.
"""

import argparse
import os
import struct
import sys
import zipfile

MASK60 = (1 << 60) - 1


def lz4_block(data, expected):
    """Decode one LZ4 block. Sequences of literals, then a match copied from earlier output."""
    out = bytearray()
    at = 0
    end = len(data)

    while at < end:
        token = data[at]
        at += 1

        length = token >> 4
        if length == 15:
            while True:
                more = data[at]
                at += 1
                length += more
                if more != 255:
                    break

        out += data[at:at + length]
        at += length

        # The last sequence stops on its literals: there is no match to follow.
        if at >= end:
            break

        offset = data[at] | (data[at + 1] << 8)
        at += 2
        if offset == 0:
            raise ValueError('bad LZ4 offset at %d' % at)

        match = token & 0x0F
        if match == 15:
            while True:
                more = data[at]
                at += 1
                match += more
                if more != 255:
                    break
        match += 4

        # Overlapping copies are legal and common - the source may run into what this loop is
        # still writing, which is how LZ4 encodes runs. Byte at a time, therefore.
        start = len(out) - offset
        if start < 0:
            raise ValueError('LZ4 offset points before the output')
        for i in range(match):
            out.append(out[start + i])

    if expected and len(out) != expected:
        print('  warning: decoded %d bytes, header said %d' % (len(out), expected), file=sys.stderr)

    return bytes(out)


def read_wni(blob):
    """Yield (hash, name) out of one .wni image held in memory."""
    if len(blob) < 18 or struct.unpack_from('<I', blob, 0)[0] != 0x20494E57:
        raise ValueError('not a WNI file')

    count, packed, unpacked = struct.unpack_from('<3I', blob, 6)
    body = lz4_block(blob[18:18 + packed], unpacked)

    at = 0
    for _ in range(count):
        if at + 8 > len(body):
            break
        key = struct.unpack_from('<Q', body, at)[0] & MASK60
        at += 8

        stop = body.find(b'\x00', at)
        if stop < 0:
            break
        yield key, body[at:stop].decode('utf-8', 'replace')
        at = stop + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+', help='.wni files, or a .zip holding them')
    ap.add_argument('--out', help='write bare names, one per line')
    ap.add_argument('--csv', help='write hash_<hex>,name')
    args = ap.parse_args()

    entries = {}
    for path in args.inputs:
        if path.lower().endswith('.zip'):
            with zipfile.ZipFile(path) as archive:
                for item in archive.infolist():
                    if not item.filename.lower().endswith('.wni'):
                        continue
                    before = len(entries)
                    try:
                        for key, name in read_wni(archive.read(item)):
                            entries[key] = name
                    except Exception as exc:
                        print('  %-34s failed: %s' % (item.filename, exc))
                        continue
                    print('  %-34s %8d entries, +%d' % (item.filename, item.file_size,
                                                        len(entries) - before))
        else:
            before = len(entries)
            with open(path, 'rb') as fh:
                for key, name in read_wni(fh.read()):
                    entries[key] = name
            print('  %-34s +%d' % (os.path.basename(path), len(entries) - before))

    print('%d distinct hashes' % len(entries))

    if args.out:
        with open(args.out, 'w', encoding='utf-8', newline='\n') as fh:
            for name in sorted(set(entries.values())):
                fh.write(name + '\n')
        print('wrote %s' % args.out)

    if args.csv:
        with open(args.csv, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('hash,name\n')
            for key in sorted(entries):
                fh.write('hash_%x,%s\n' % (key, entries[key]))
        print('wrote %s' % args.csv)


if __name__ == '__main__':
    main()
