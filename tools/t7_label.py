#!/usr/bin/env python3
"""Give the unrecoverable hashes a readable label instead of a hex placeholder.

    python tools/t7_label.py --scripts decompiled/ --known bo3.csv --out labels.csv

Why this and not more searching. Every source that can name a Black Ops 3 script hash has now been
tried and measured: published lists, decompiled scripts, the older games' plaintext, Black Ops 4's
leaked source, the mod tools, every non-script asset in all 264 zones, GPU sweeps, mutation of
sibling names, per-hash context search. What is left - some twenty thousand hashes - is unrecoverable
in the strict sense, because a 32-bit hash cannot arbitrate a search wide enough to reach it and no
surviving artefact spells the names out.

But `var_1a2b3c4d` is not unreadable because the name is missing. It is unreadable because the
placeholder throws away everything the code *does* say about it, which is a great deal:

    what it is       a field, an event, a function, a table key
    where it lives   the namespace or file that uses it most
    what it concerns the known identifiers that keep appearing beside it

So this writes `x_evt_zm_castle_dragon_3f2a` where the decompiler wrote `hash_9f3c3f2a`. Nothing is
guessed - every part of that label is read off the code - and the result is a decompilation a person
can follow, which is the thing the names were wanted for in the first place.

Three rules keep it honest, and they are the whole design:

  the `x_` prefix   a label is never mistakable for a recovered name, in a diff or in a table
  the hex suffix    the original hash is still there, so a label is reversible and auditable, and
                    two labels can never collide however similar their context
  a separate file   labels never enter the verified table. Anything claiming to be a name in this
                    repository recomputes to its hash; a label does not, and must not pretend to.

If a real name turns up later, it simply supersedes the label - the hash is right there in it.
"""

import argparse
import collections
import math
import os
import re
import sys

HASHED = re.compile(r'\b(var|function|namespace|hash|event|class|method)_([0-9a-fA-F]{6,16})\b')
NAMESPACE = re.compile(r'^\s*#namespace\s+([a-z_][a-z0-9_]*)\s*;', re.M)
IDENT = re.compile(r'\b([a-z_][a-z0-9_]{2,40})\b')
PLACEHOLDER = re.compile(r'^(?:var|function|namespace|hash|event|class|method)_[0-9a-f]{6,16}$')

# What a hash is, read off the syntax it sits in. Ordered: the first that matches wins, so the
# specific forms are listed before the general ones.
ROLES = [
    ('evt', re.compile(r'(?:notify|waittill|endon|waittill_any|waittill_timeout)\s*\(\s*[^)]*#?"?%s')),
    ('fld', re.compile(r'(?:self|level|game)\s*\.\s*%s\b')),
    ('key', re.compile(r'\[\s*#?"?%s"?\s*\]')),
    ('fn', re.compile(r'%s\s*\(')),
    ('ns', re.compile(r'%s\s*::')),
]

# Words that sit beside everything and so distinguish nothing.
STOP = {
    'self', 'level', 'game', 'else', 'return', 'thread', 'true', 'false', 'undefined', 'isdefined',
    'function', 'var', 'foreach', 'while', 'break', 'continue', 'switch', 'case', 'default',
    'wait', 'waittill', 'notify', 'endon', 'size', 'params', 'checksum', 'offset', 'namespace',
    'private', 'autoexec', 'const', 'int', 'float', 'string', 'array', 'struct', 'value',
}


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
    """Every unnamed hash, with the evidence the label will be built from."""
    seen = collections.defaultdict(lambda: {
        'kinds': collections.Counter(), 'roles': collections.Counter(),
        'homes': collections.Counter(), 'near': collections.Counter(), 'uses': 0})
    document_freq = collections.Counter()
    files_seen = 0

    for base, _dirs, files in os.walk(root):
        for name in files:
            if not name.endswith(('.gsc', '.csc')):
                continue
            files_seen += 1

            with open(os.path.join(base, name), encoding='utf-8', errors='replace') as fh:
                text = fh.read()

            match = NAMESPACE.search(text)
            home = match.group(1) if match else ''
            if not home or PLACEHOLDER.match(home):
                home = os.path.splitext(name)[0]

            words = {w for w in IDENT.findall(text) if not PLACEHOLDER.match(w)}
            document_freq.update(words)

            lines = text.split('\n')
            for number, line in enumerate(lines):
                for kind, digest in HASHED.findall(line):
                    value = int(digest, 16)
                    if value in known:
                        continue

                    entry = seen[value]
                    entry['uses'] += 1
                    entry['kinds'][kind] += 1
                    entry['homes'][home] += 1

                    token = '%s_%s' % (kind, digest)
                    for role, pattern in ROLES:
                        if re.search(pattern.pattern % re.escape(token), line):
                            entry['roles'][role] += 1
                            break

                    # One line either side: near enough to be about the same thing.
                    for other in lines[max(0, number - 1):number + 2]:
                        for word in IDENT.findall(other):
                            if word not in STOP and not PLACEHOLDER.match(word):
                                entry['near'][word] += 1

    return seen, document_freq, files_seen


def topic(near, document_freq, files_seen):
    """The most telling word beside this hash: frequent here, rare everywhere else."""
    best, score = None, 0.0
    for word, count in near.items():
        weight = count * math.log(files_seen / (1.0 + document_freq.get(word, 0)))
        if weight > score:
            best, score = word, weight
    return best


def label_of(value, entry, document_freq, files_seen, parts):
    pieces = ['x']

    if entry['roles']:
        pieces.append(entry['roles'].most_common(1)[0][0])
    else:
        pieces.append({'var': 'fld', 'function': 'fn', 'namespace': 'ns',
                       'hash': 'evt', 'method': 'fn'}.get(entry['kinds'].most_common(1)[0][0], 'x'))

    if parts >= 3:
        pieces.append(entry['homes'].most_common(1)[0][0])

    if parts >= 4:
        word = topic(entry['near'], document_freq, files_seen)
        if word:
            pieces.append(word)

    # The hash keeps the label unique and reversible, whatever the context turned out to be.
    pieces.append('%08x' % value)
    clean = []
    for piece in pieces:
        piece = re.sub(r'[^a-z0-9_]+', '', piece.lower()).strip('_')
        piece = piece[:28].strip('_')
        clean.append(piece if piece else 'x')
    return '_'.join(clean)[:96]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scripts', required=True, help='directory of decompiled scripts')
    ap.add_argument('--known', nargs='+', required=True, help='verified hash,name lists')
    ap.add_argument('--out', required=True)
    ap.add_argument('--parts', type=int, default=4, choices=(2, 3, 4),
                    help='how much context goes in a label: role only, plus home, plus topic (4)')
    ap.add_argument('--min-uses', type=int, default=1)
    args = ap.parse_args()

    known = read_known(args.known)
    seen, document_freq, files_seen = scan(args.scripts, known)

    targets = {h: e for h, e in seen.items() if e['uses'] >= args.min_uses}
    print('unnamed hashes         : %d' % len(seen))
    print('labelled               : %d' % len(targets))

    labels = {}
    for value, entry in targets.items():
        labels[value] = label_of(value, entry, document_freq, files_seen, args.parts)

    assert len(set(labels.values())) == len(labels), 'labels must be unique'

    roles = collections.Counter(l.split('_')[1] for l in labels.values())
    print('by role                : %s'
          % ', '.join('%s %d' % kv for kv in roles.most_common()))

    with open(args.out, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('hash,name\n')
        for value in sorted(labels):
            fh.write('%08x,%s\n' % (value, labels[value]))
    print('wrote %s' % args.out)

    print()
    print('a label is not a name - it is what the code says about a name nobody can recover:')
    for value in sorted(targets, key=lambda h: -targets[h]['uses'])[:12]:
        print('  %-9s -> %s' % ('%08x' % value, labels[value]))

    return 0


if __name__ == '__main__':
    sys.exit(main())
