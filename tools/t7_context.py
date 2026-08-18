#!/usr/bin/env python3
"""Rank the script hashes still unnamed, and show what surrounds each one.

    python tools/t7_context.py --scripts decompiled/ --known bo3.csv --out report.txt

Why ranking, and why context. After every published list and every decompilation has been folded
in, Black Ops 3 leaves tens of thousands of hashes with no name, and they are not equally worth
attention: most appear once, as a local variable in one function, and naming them changes nothing.
A few hundred appear everywhere, and those are the ones holding the map together.

The hash cannot settle them. Thirty-two bits stays precise only while the targets are few - a
hundred false matches buys 4.8e9 candidates against 90 targets and 1.5e7 against 28 400 - so a
blanket sweep returns noise. Working a small batch at a time keeps the budget, which is why this
sorts rather than dumps.

And what makes a batch workable is the surrounding code. A hash used as `self.var_X[ #"hash_Y" ]`
is an array key; one appearing as a function parameter has a type its callers reveal; one in
`level notify( #"hash_Z" )` is an event name and the listeners say what happened. Anchors like
those cut the search from open-ended to a middle of a few characters, and the arithmetic there is
kind: with a known prefix and suffix, an unknown middle of six characters or fewer has about 0.6
expected candidates over the whole 32-bit space. One answer, not a shortlist.
"""

import argparse
import collections
import os
import re

PATTERN = re.compile(r'\b(var|function|namespace|hash|event|class|method)_([0-9a-fA-F]{6,16})\b')


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


def scan(root, known):
    """Every unnamed hash, with where it appears and what sits around it."""
    seen = collections.defaultdict(
        lambda: {'count': 0, 'kinds': collections.Counter(), 'files': set(), 'lines': []})

    for base, _dirs, files in os.walk(root):
        for name in files:
            if not name.endswith(('.gsc', '.csc')):
                continue

            path = os.path.join(base, name)
            with open(path, encoding='utf-8', errors='replace') as fh:
                lines = fh.read().split('\n')

            for number, line in enumerate(lines):
                for kind, digest in PATTERN.findall(line):
                    value = int(digest, 16)
                    if value in known:
                        continue

                    entry = seen[value]
                    entry['count'] += 1
                    entry['kinds'][kind] += 1
                    entry['files'].add(name)
                    if len(entry['lines']) < 6:
                        entry['lines'].append('%s:%d  %s' % (name, number + 1, line.strip()[:120]))

    return seen


def anchors(lines, digest):
    """Names sitting next to this hash, which are what a targeted search can hang on.

    A hash rarely appears alone. It is a key into an array whose own name is known, or a parameter
    beside other parameters, or one of a run of sibling constants. Those neighbours carry the
    naming convention of whatever the unknown belongs to, so they are the first thing to read when
    guessing at it.
    """
    near = collections.Counter()
    token = re.compile(r'\b([a-z][a-z0-9_]{3,40})\b')
    for line in lines:
        for word in token.findall(line):
            if word.startswith(('var_', 'function_', 'hash_', 'namespace_')):
                continue
            if word in ('isdefined', 'self', 'level', 'else', 'return', 'thread', 'true', 'false'):
                continue
            near[word] += 1

    return [w for w, _ in near.most_common(6)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scripts', required=True, help='directory of decompiled scripts')
    ap.add_argument('--known', nargs='+', required=True, help='hash,name lists already resolved')
    ap.add_argument('--out', help='write the ranked report')
    ap.add_argument('--top', type=int, default=200)
    ap.add_argument('--min-count', type=int, default=5)
    args = ap.parse_args()

    known = read_known(args.known)
    seen = scan(args.scripts, known)

    print('unnamed hashes: %d' % len(seen))
    print('appearing more than %d times: %d'
          % (args.min_count, sum(1 for e in seen.values() if e['count'] > args.min_count)))

    ranked = sorted(seen.items(), key=lambda kv: (-kv[1]['count'], kv[0]))
    ranked = [r for r in ranked if r[1]['count'] > args.min_count][:args.top]

    out = open(args.out, 'w', encoding='utf-8', newline='\n') if args.out else None

    def emit(text=''):
        if out:
            out.write(text + '\n')
        else:
            print(text)

    emit('# %d hashes worth naming, most used first' % len(ranked))
    emit('# Each block: the hash, how it is written, how often, in how many files,')
    emit('# the names sitting next to it, then where it appears.')
    emit()

    for value, entry in ranked:
        kind = entry['kinds'].most_common(1)[0][0]
        emit('%08x  %-9s  %d uses in %d file(s)' % (value, kind, entry['count'], len(entry['files'])))
        close = anchors(entry['lines'], value)
        if close:
            emit('    near: %s' % ', '.join(close))
        for line in entry['lines']:
            emit('    %s' % line)
        emit()

    if out:
        out.close()
        print('wrote %s' % args.out)


if __name__ == '__main__':
    main()
