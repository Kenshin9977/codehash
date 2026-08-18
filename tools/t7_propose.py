#!/usr/bin/env python3
"""Propose names for unresolved script hashes by mutating the resolved ones beside them.

    python tools/t7_propose.py --scripts decompiled/ --known bo3.csv --out proposed.csv

A different move from sweeping a dictionary, and a better-aimed one. A dictionary sweep asks what
strings exist; this asks what a programmer would have written in this file, given what they wrote
on the lines around it.

Identifiers come in families. A file that contains `port_wave1` and `port_wave2` almost certainly
contains `port_wave3`; one with `zombie_spawn_init` probably has `zombie_spawn_stop`. The resolved
names in a file are therefore a template for its unresolved ones, and mutating them generates a
few thousand candidates where a dictionary would generate billions - which matters, because a
32-bit hash only stays trustworthy while the candidate count is small.

Verification is free and exact: hash the proposal and see. So there is no risk in proposing badly,
only wasted work, and every mutation family below can be measured against how many names it
actually recovers.
"""

import argparse
import collections
import itertools
import os
import re

MASK32 = 0xFFFFFFFF
SEED = 0x4B9ACE2F
PRIME = 0x1000193

HASHED = re.compile(r'\b(?:var|function|namespace|hash|event|class|method)_([0-9a-fA-F]{6,16})\b')
IDENT = re.compile(r'\b([a-z_][a-z0-9_]{2,48})\b')
PLACEHOLDER = re.compile(r'^(?:var|function|namespace|hash|event|class|method)_[0-9a-f]{6,16}$')
TRAILING = re.compile(r'^(.*?)(\d+)$')


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


def mutations(names, tokens):
    """Candidate names built out of what this file already says.

    Four families, each answering a way a sibling identifier tends to differ from its neighbours:

      counting   a trailing number moved through a small range, which catches wave1/wave2/wave3
                 and the stage, part and index series that fill this codebase
      swapping   one token replaced by another the file uses, which catches init/stop/reset pairs
                 and the colour, side and state variants of a single thing
      trimming   a leading or trailing token dropped, since a family often has a bare member
      joining    the head of one local name with the tail of another
    """
    out = set()

    for name in names:
        parts = name.split('_')

        match = TRAILING.match(name)
        if match:
            stem, number = match.group(1), int(match.group(2))
            for n in range(max(0, number - 4), number + 6):
                out.add('%s%d' % (stem, n))
        else:
            for n in range(0, 5):
                out.add('%s%d' % (name, n))
                out.add('%s_%d' % (name, n))

        for i in range(len(parts)):
            for token in tokens:
                if token != parts[i]:
                    out.add('_'.join(parts[:i] + [token] + parts[i + 1:]))

        if len(parts) > 1:
            out.add('_'.join(parts[1:]))
            out.add('_'.join(parts[:-1]))

    heads = {n.split('_')[0] for n in names if '_' in n}
    tails = {n.split('_', 1)[1] for n in names if '_' in n}
    for head, tail in itertools.product(heads, tails):
        out.add(head + '_' + tail)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scripts', required=True)
    ap.add_argument('--known', nargs='+', required=True)
    ap.add_argument('--out')
    ap.add_argument('--tokens', type=int, default=60,
                    help='how many of a file\'s commonest tokens may be substituted in')
    args = ap.parse_args()

    known = read_known(args.known)
    found = {}
    tried = 0
    families = collections.Counter()

    for base, _dirs, files in os.walk(args.scripts):
        for name in files:
            if not name.endswith(('.gsc', '.csc')):
                continue

            with open(os.path.join(base, name), encoding='utf-8', errors='replace') as fh:
                text = fh.read()

            targets = {int(h, 16) for h in HASHED.findall(text)} - known
            if not targets:
                continue

            # The resolved identifiers of this file are the template; its token frequencies say
            # which substitutions are idiomatic here rather than idiomatic in general.
            local = set()
            counter = collections.Counter()
            for word in IDENT.findall(text):
                if PLACEHOLDER.match(word):
                    continue
                local.add(word)
                for token in word.split('_'):
                    if token:
                        counter[token] += 1

            tokens = [t for t, _ in counter.most_common(args.tokens)]
            for candidate in mutations(local, tokens):
                if not candidate or len(candidate) > 64:
                    continue
                tried += 1
                digest = t7_hash(candidate)
                if digest in targets and digest not in found:
                    found[digest] = candidate
                    families[candidate.count('_')] += 1

    print('candidates tried: %.3g' % tried)
    print('names recovered : %d' % len(found))
    print('by token count  : %s' % dict(sorted(families.items())))

    if args.out:
        with open(args.out, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('hash,name\n')
            for digest in sorted(found):
                fh.write('%08x,%s\n' % (digest, found[digest]))
        print('wrote %s' % args.out)


if __name__ == '__main__':
    main()
