#!/usr/bin/env python3
"""Recompute the hash of every recovered name and check it against the target list.

Never skip this. A dictionary search over a 60-bit space returns false positives by construction -
roughly `candidates * targets / 2^60` of them - and the search reports a hit the moment a bitmap
and a table agree, which is exactly the kind of agreement a truncated hash can fake. Recomputing
costs nothing next to a ten hour run and it is the only thing that turns a list of guesses into a
list of names.

    python tools/verify_names.py --targets targets.txt --names found.txt [--csv out.csv]

Both files are one entry per line. Targets may be bare hex or `hash_<hex>`; names are the strings
the search produced. Exit status is non-zero if anything failed to verify, so this drops straight
into a pipeline.
"""

import argparse
import collections
import sys

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = 0xFFFFFFFFFFFFFFFF
MASK60 = (1 << 60) - 1


def fnv60(text):
    """The games' hash: FNV-1a over the bytes, kept to its low 60 bits."""
    value = FNV_OFFSET
    for byte in text.encode('utf-8'):
        value = ((value ^ byte) * FNV_PRIME) & MASK64

    return value & MASK60


def read_targets(path):
    targets = set()
    for line in open(path, encoding='utf-8'):
        line = line.strip().lower()
        if not line:
            continue
        if line.startswith('hash_'):
            line = line[5:]
        if line.startswith('0x'):
            line = line[2:]
        try:
            targets.add(int(line, 16) & MASK60)
        except ValueError:
            pass

    return targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', required=True, help='hashes the search was aimed at')
    ap.add_argument('--names', required=True, help='names the search produced')
    ap.add_argument('--csv', help='write hash,name for everything that verified')
    ap.add_argument('--slash', action='store_true',
                    help='normalise backslashes to forward slashes before hashing, which is what '
                         'the games do to asset paths - without it every path-shaped name fails')
    args = ap.parse_args()

    targets = read_targets(args.targets)
    names = [line.rstrip('\n') for line in open(args.names, encoding='utf-8') if line.strip()]
    if args.slash:
        names = [n.replace(chr(92), '/') for n in names]

    hit = collections.defaultdict(list)
    missed = []
    for name in names:
        digest = fnv60(name)
        if digest in targets:
            hit[digest].append(name)
        else:
            missed.append(name)

    print('targets      %d' % len(targets))
    print('names        %d' % len(names))
    print('verified     %d' % sum(len(v) for v in hit.values()))
    print('unverified   %d' % len(missed))
    print('targets hit  %d (%.2f%%)' % (len(hit), 100.0 * len(hit) / (len(targets) or 1)))

    # Two names on one hash is a real 60-bit collision, and only one of them is the asset. Worth
    # calling out rather than silently keeping both: the wrong one is indistinguishable by hash
    # and has to be settled by looking at what it names.
    clashes = {h: v for h, v in hit.items() if len(v) > 1}
    if clashes:
        print('\n%d hash(es) claimed by more than one name - pick by hand:' % len(clashes))
        for digest, found in sorted(clashes.items()):
            print('  %012x  %s' % (digest, '  |  '.join(sorted(found))))

    if missed:
        print('\nfirst unverified names:')
        for name in missed[:10]:
            print('  %s' % name)

    if args.csv:
        # Every name goes in, verified or not, with a column saying which. Writing only the
        # survivors throws away what you most want when a run disappoints - what it proposed and
        # how far off it was - and it makes two runs impossible to diff.
        with open(args.csv, 'w', encoding='utf-8', newline='\n') as out:
            out.write('hash,name,verified\n')
            for digest in sorted(hit):
                for name in sorted(hit[digest]):
                    out.write('hash_%x,%s,yes\n' % (digest, name))
            for name in sorted(missed):
                out.write('hash_%x,%s,no\n' % (fnv60(name), name))
        print('\nwrote %s' % args.csv)

    return 1 if missed else 0


if __name__ == '__main__':
    sys.exit(main())
