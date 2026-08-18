#!/usr/bin/env python3
"""Name Black Ops 3 script hashes from the game's own non-script files.

    python tools/t7_from_rawfiles.py --files extracted/ --known bo3.csv --out fromraw.csv

The idea, and why it is not obvious. Everything else in this repository looks for names inside
code - published lists, decompiled scripts, sibling identifiers. But a script hash is not always
the name of something the script declares. Very often it is the name of something *another asset*
declares and the script merely refers to:

    level waittill( #"hash_a1b2c3d4" );

is a script waiting on an event that a cinematic scene file raises. The script never spells the
name; the scene does. Same for the string tables that drive challenges, the vision files, the
gamedata sheets, the sound aliases. The declaring side ships as plain text and was never hashed.

So this reads the extracted rawfiles rather than the scripts, and pulls candidates in five shapes,
because the declaring file's format decides how a name is written down:

    identifiers     bare words, the common case
    quoted          string literals inside XML, JSON and config text
    paths           hashed whole, separators and all, plus their basenames both with and
                    without an extension - a script may refer to either
    cells           one comma- or tab-separated field, taken whole: string tables and gamedata
                    sheets put a whole name in a cell, spaces and capitals included
    runs            a whole printable run, taken as-is, which is how single-column lists and
                    binary constant tables both write a name down

Reading is done on bytes, not on decoded text, and this is the part that matters most. Two of the
three largest groups the zones give up are compiled formats - Lua bytecode and compiled GSC - and
both keep their string constants in the clear inside a binary container. Splitting the bytes into
printable runs recovers those constants and, on a genuinely textual file, gives exactly the same
thing a line split would, since a newline is not printable. One reader, both cases.

The measurement that justifies it: over the rawfiles of the 264 zones this recovers names that no
amount of code reading would have produced - `hendricks_disappear`, `safehouse_explosion_igc_done`,
`start_fade_to_white`, `water_drain_complete`. They are event names, and events are exactly the
category whose declaration lives outside the code.

False matches must be counted, though, and this is where a run over the whole game gets caught.
Expected false matches are `candidates * targets / 2^32`, so 1.55 million distinct strings against
21 614 unnamed hashes expects **7.8** of them - and the raw run returned 16 hits of which exactly
seven were `PfF`, `lzS`, `*PnrE`, `AZoLBm`, `NOX`, `sff`, `*sJS`. Short random-looking runs out of
the middle of compiled Lua and GSC, matching by accident, and indistinguishable from an answer if
you only look at the hash.

Hence `--min-length`, which is the whole defence: coincidence is uniform over the corpus, and the
corpus's short mixed-case runs are where it lands. Requiring eight characters or an underscore
throws away the accidents without touching a single real name, because a name still unresolved
after every published list and every decompilation is never three characters long.
"""

import argparse
import collections
import os
import re
import sys

MASK32 = 0xFFFFFFFF
SEED = 0x4B9ACE2F
PRIME = 0x1000193

IDENT = re.compile(r'[A-Za-z_][A-Za-z0-9_]{2,63}')
QUOTED = re.compile(r'"([^"\n]{3,64})"')
PATH = re.compile(r'[A-Za-z0-9_.\-]+(?:[\\/][A-Za-z0-9_.\-]+)+')
PLACEHOLDER = re.compile(r'^(?:var|function|namespace|hash|event|class|method)_[0-9a-f]{6,16}$')
# A short candidate is only allowed through if it is structured - a compound name or a path.
SEPARATED = re.compile(r'[_/\\]')
# A maximal run of printable bytes. On text this splits at newlines and so yields lines; on a
# compiled Lua or GSC file it yields the entries of the constant table.
RUN = re.compile(rb'[\x20-\x7e]{3,96}')

# Anything that is plainly not text, or is so large that reading it whole costs more than the
# names it could hold. Scripts are excluded on purpose: they are harvested by t7_hash.py, and
# counting them here would hide how much the non-script files are actually worth.
SKIP = ('.gsc', '.csc', '.gsh', '.png', '.dds', '.jpg', '.wav', '.flac', '.bik', '.ogg')


def t7_hash(text):
    value = SEED
    for char in text.lower():
        value = ((ord(char) ^ value) * PRIME) & MASK32
    return (value * PRIME) & MASK32


def read_known(paths):
    known = set()
    for path in paths:
        with open(path, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                digest = line.split(',', 1)[0].strip().lower().replace('hash_', '')
                try:
                    known.add(int(digest, 16))
                except ValueError:
                    pass
    return known


def strings_of(data, kinds):
    """Every candidate name one file's bytes can offer, in each shape the caller asked for."""
    out = set()

    for match in RUN.finditer(data):
        run = match.group().decode('ascii')

        if 'run' in kinds and len(run) <= 64:
            out.add(run)
            out.add(run.strip())

        if 'ident' in kinds:
            out.update(IDENT.findall(run))

        if 'quoted' in kinds:
            out.update(QUOTED.findall(run))

        if 'path' in kinds:
            for path in PATH.findall(run):
                out.add(path)
                # A script may name the file, the file without its extension, or the whole path.
                base = re.split(r'[\\/]', path)[-1]
                out.add(base)
                out.add(os.path.splitext(base)[0])

        if 'cell' in kinds and (',' in run or '	' in run):
            for cell in re.split(r'[,	]', run):
                cell = cell.strip().strip('"')
                if cell:
                    out.add(cell)

    return {s for s in out if 3 <= len(s) <= 64 and not PLACEHOLDER.match(s.lower())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True,
                    help='directories of extracted non-script assets')
    ap.add_argument('--known', nargs='+', required=True, help='hash,name lists already resolved')
    ap.add_argument('--targets', help='hashes to solve, one per line; default is every hash '
                                      'not in --known that appears in --scripts')
    ap.add_argument('--scripts', help='decompiled scripts, to take the target list from')
    ap.add_argument('--kinds', default='ident,quoted,path,cell,run')
    ap.add_argument('--min-length', type=int, default=8,
                    help='shortest candidate allowed unless it carries a separator (8). This is '
                         'what keeps coincidence out - see the note at the top of the file')
    ap.add_argument('--out')
    args = ap.parse_args()

    kinds = set(args.kinds.split(','))
    known = read_known(args.known)
    print('names already resolved: %d' % len(known))

    targets = set()
    if args.targets:
        for line in open(args.targets, encoding='utf-8', errors='replace'):
            try:
                targets.add(int(line.split(',', 1)[0].strip().lower().replace('hash_', ''), 16))
            except (ValueError, IndexError):
                pass
    if args.scripts:
        pattern = re.compile(
            r'\b(?:var|function|namespace|hash|event|class|method)_([0-9a-fA-F]{6,16})\b')
        for base, _dirs, files in os.walk(args.scripts):
            for name in files:
                if name.endswith(('.gsc', '.csc')):
                    with open(os.path.join(base, name), encoding='utf-8', errors='replace') as fh:
                        targets.update(int(h, 16) for h in pattern.findall(fh.read()))
    targets -= known
    if not targets:
        ap.error('no targets: pass --targets or --scripts')
    print('hashes to solve       : %d' % len(targets))

    # Where a name came from is worth keeping: it says which asset types are worth extracting
    # next time, and it is the only evidence that a recovered name is not a coincidence.
    found = {}
    source = {}
    seen = set()
    tested = set()
    scanned = 0

    for root in args.files:
        for base, _dirs, files in os.walk(root):
            for name in files:
                if name.lower().endswith(SKIP):
                    continue
                path = os.path.join(base, name)
                try:
                    if os.path.getsize(path) > 64 * 1024 * 1024:
                        continue
                    with open(path, 'rb') as fh:
                        data = fh.read()
                except OSError:
                    continue

                scanned += 1
                for candidate in strings_of(data, kinds):
                    if candidate in seen:
                        continue
                    seen.add(candidate)
                    if len(candidate) < args.min_length and not SEPARATED.search(candidate):
                        continue
                    tested.add(candidate)
                    digest = t7_hash(candidate)
                    if digest in targets and digest not in found:
                        found[digest] = candidate
                        source[digest] = os.path.relpath(path, root)

    print('files read            : %d' % scanned)
    print('distinct strings      : %d, of which %d pass --min-length %d'
          % (len(seen), len(tested), args.min_length))
    print('false matches expected: %.1f' % (len(tested) * len(targets) / 2.0 ** 32))
    print('names recovered       : %d  (%.2f%% of the targets)'
          % (len(found), 100.0 * len(found) / max(1, len(targets))))

    kinds_of = collections.Counter(os.path.splitext(p)[1].lower() or '(none)'
                                   for p in source.values())
    print('by declaring file type: %s'
          % ', '.join('%s %d' % kv for kv in kinds_of.most_common(12)))

    for digest, name in found.items():
        assert t7_hash(name) == digest

    if args.out:
        with open(args.out, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('hash,name,source\n')
            for digest in sorted(found):
                fh.write('%08x,%s,%s\n' % (digest, found[digest], source[digest]))
        print('wrote %s' % args.out)

    return 0


if __name__ == '__main__':
    sys.exit(main())
