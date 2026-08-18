#!/usr/bin/env python3
"""Black Ops 3 script hashes: the algorithm, and the only way to resolve them that works.

    python tools/t7_hash.py --harvest t7-source.zip --known hashes-scr-bo3.csv --out bo3.csv
    python tools/t7_hash.py --hash player_is_in_laststand

Black Ops 3 hashes its script identifiers with Treyarch's canonical hash, which is FNV-1a 32-bit
with a different seed and one extra round at the end:

    h = 0x4B9ACE2F
    for each character, lowercased:  h = (c XOR h) * 0x1000193
    h = h * 0x1000193

Recovered by testing candidates against 2 000 published pairs - 1999 matched, and the one that did
not is a casing difference. Independently confirmed against a second list where all 1 071 pairs
recompute.

**These cannot be cracked, and that is not a matter of effort.** Thirty-two bits is too short to
confirm anything: a search the size of the one this repository runs against Black Ops 4 would
return 3.5e8 false matches against a mere thousand 32-bit targets, every one indistinguishable
from a right answer. The hash stops being evidence.

So they are harvested, not cracked, which is what Cerberus does - its table was built from every
GSC source its author could find across CoD 4, World at War, Black Ops 1 and Black Ops 3. The
measurement here says how well that works: identifiers taken out of 2 226 decompiled Black Ops 3
scripts reproduce **95.8%** of the 31 190 hashes the community has published. Almost everything
worth naming is already written down somewhere, in a game whose scripts were never hashed or in a
decompilation of one whose were.

Where it stops. Decompiling every script-bearing zone of Black Ops 3 - 106 of its 264 fastfiles,
2 438 scripts - leaves 79 209 hashes that acts cannot name. This table names 64.1% of them against
the community list's 34.7%, and 28 400 stay unnamed.

World at War and Black Ops 1 shipped their scripts in the clear, and Zombie Chronicles remasters
their maps, so those looked like the obvious next source. They are worth 408 names, 1.4% of what
is left. The reason is worth keeping: what the older games share with Black Ops 3 was already in
the Black Ops 3 decompilation this table came from. Old sources only add what disappeared on the
way - notify strings with spaces in them, mostly, which no decompiler reconstructs.

What remains cannot be swept either. A 32-bit search stays precise only while the targets are few:
one hundred false matches allows 4.8e9 candidates against 90 targets and 1.5e7 against 28 400.
That is the exact opposite of the 60-bit case, where more targets helped. Past a few thousand
targets the hash stops arbitrating and the surrounding code is the only judge left.
"""

import argparse
import collections
import os
import re
import sys
import zipfile

MASK32 = 0xFFFFFFFF
SEED = 0x4B9ACE2F
PRIME = 0x1000193


def t7_hash(text):
    """Treyarch canonical hash, as Black Ops 3 applies it to script identifiers."""
    value = SEED
    for char in text.lower():
        value = ((ord(char) ^ value) * PRIME) & MASK32

    return (value * PRIME) & MASK32


# What a decompiler writes when it does not know: var_49dbff, function_ef0ce9fb, namespace_36e5.
# These hash to something and so pass verification, but they are placeholders, not names, and a
# table full of them is worse than useless - it answers a hash with the fact that it is unknown.
PLACEHOLDER = re.compile(r'^(?:var|function|namespace|hash|event|class|method)_[0-9a-f]{6,16}$')


def identifiers(text):
    """Everything in a script that might be hashed: names, string literals, script paths."""
    found = set()
    found.update(m.lower() for m in re.findall(r'[A-Za-z_][A-Za-z0-9_]{2,63}', text))
    found.update(m.lower() for m in re.findall(r'"([^"\n]{3,64})"', text))
    # A script path is hashed whole, separators and all.
    found.update(m.lower() for m in re.findall(r'[A-Za-z0-9_]+(?:[\\/][A-Za-z0-9_]+)+', text))
    return {f for f in found if not PLACEHOLDER.match(f)}


def harvest(paths):
    words = set()
    for path in paths:
        if path.lower().endswith('.zip'):
            with zipfile.ZipFile(path) as archive:
                for item in archive.infolist():
                    if item.filename.lower().endswith(('.gsc', '.csc')):
                        words |= identifiers(archive.read(item).decode('utf-8', 'replace'))
        elif os.path.isdir(path):
            for root, _dirs, files in os.walk(path):
                for name in files:
                    if name.lower().endswith(('.gsc', '.csc')):
                        with open(os.path.join(root, name), encoding='utf-8',
                                  errors='replace') as fh:
                            words |= identifiers(fh.read())
        else:
            with open(path, encoding='utf-8', errors='replace') as fh:
                words |= identifiers(fh.read())

    return words


def read_pairs(path):
    pairs = {}
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            parts = line.rstrip('\n').split(',')
            if len(parts) < 2:
                continue
            digest = parts[0].strip().lower().replace('hash_', '')
            try:
                pairs[int(digest, 16)] = parts[1].strip()
            except ValueError:
                pass
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hash', help='print the hash of one string and stop')
    ap.add_argument('--harvest', nargs='*', default=[],
                    help='script sources: a .zip, a directory, or files')
    ap.add_argument('--known', nargs='*', default=[],
                    help='hash,name lists to merge in and to score the harvest against')
    ap.add_argument('--out', help='write the merged table')
    args = ap.parse_args()

    if args.hash:
        print('%08x  %s' % (t7_hash(args.hash), args.hash))
        return 0

    if not args.harvest and not args.known:
        ap.error('nothing to do: pass --hash, --harvest or --known')

    table = {}
    for path in args.known:
        before = len(table)
        kept = 0
        for digest, name in read_pairs(path).items():
            # Only entries that recompute go in. A published list can carry a wrong casing or a
            # different variant, and a name table nobody can verify is worth less than none.
            if t7_hash(name) == digest and not PLACEHOLDER.match(name.lower()):
                table[digest] = name
                kept += 1
        print('  %-28s %6d verified, +%d new' % (os.path.basename(path), kept,
                                                 len(table) - before))

    if args.harvest:
        words = harvest(args.harvest)
        print('harvested %d distinct identifiers' % len(words))

        published = set(table)
        mine = collections.OrderedDict()
        for word in sorted(words):
            mine.setdefault(t7_hash(word), word)

        if published:
            covered = len(published & set(mine))
            print('reproduces %d of %d published hashes (%.1f%%)'
                  % (covered, len(published), 100.0 * covered / len(published)))

        before = len(table)
        for digest, name in mine.items():
            table.setdefault(digest, name)
        print('  %-28s %6d hashes, +%d new' % ('harvest', len(mine), len(table) - before))

    print('table: %d hash to name pairs, every one recomputed' % len(table))

    if args.out:
        with open(args.out, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('hash,name\n')
            for digest in sorted(table):
                fh.write('%08x,%s\n' % (digest, table[digest]))
        print('wrote %s' % args.out)

    return 0


if __name__ == '__main__':
    sys.exit(main())
