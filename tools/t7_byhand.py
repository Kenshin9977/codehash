#!/usr/bin/env python3
"""Name one hash at a time from the lines it appears on.

    python tools/t7_byhand.py --scripts decompiled/ --known bo3.csv --out named.csv

The last thing that works, and the reason it works is a budget.

Expected false matches are `candidates * targets / 2^32`. Sweeping 25 000 targets at once allows
1.5e7 candidates before the noise takes over - nothing. Sweeping ONE target allows 4.3e11 for the
same hundred false matches, which is more than any sensible candidate set will ever hold. The
hash goes from useless to decisive purely by asking about one thing at a time.

So this works per hash, and takes its vocabulary from the narrowest place available: the lines
that hash actually appears on, plus the function it sits in. Twenty or thirty tokens, not forty
thousand. `function_ce931b57( "zmb_pickup_nurgle", 1 )` says the answer involves pickups and
sounds and registration, and nothing about vehicles or weather.

The measurements that justify the shape: a global two-token sweep names 19% of held-out hashes at
97.5% precision, a file-local one 6.7% at 100%, and the prefix-times-word sweep 82.4% - its errors
being things like `update_zombie_gravity_dog` for `enableterrainik`, which look exactly like real
identifiers and cannot be caught by reading. Narrower is not just safer here, it is the only thing
that stays trustworthy.

And now the result that matters most, because it is negative. Run over the 4 048 hashes used five
times or more, this recovers **one**. The reason is measurable: taking hashes whose names are
already known and asking whether those names' words appear in the lines around them,

    no word of the name present    91.7%
    some of them                    6.7%
    all of them                     1.6%

Context says what a thing does, not what it is called. `function_ce931b57( "zmb_pickup_nurgle", 1 )`
is obviously registering a pickup sound and gives away nothing about whether the author wrote
register_sound, add_alias or something with no word in common with either. Reasoning it out by hand
fails for the same reason: 23 195 candidates written against eight of the most-used hashes, chosen
after reading their code, returned zero.

So this tool is kept for what it is - the narrowest, safest search available - and not as an answer.
What is left needs source material nobody has published yet, or someone who already knows the
codebase by name.
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
QUOTED = re.compile(r'"([a-z0-9_]{3,48})"')
PLACEHOLDER = re.compile(r'^(?:var|function|namespace|hash|event|class|method)_[0-9a-f]{6,16}$')
FUNC = re.compile(r'^\s*(?:autoexec\s+|private\s+)*function\s+(?:autoexec\s+|private\s+)*([a-z_][a-z0-9_]*)')


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


def context_tokens(lines, window):
    """Tokens from the lines around one appearance, and from the function heading it."""
    tokens = collections.Counter()
    for line in lines:
        for word in IDENT.findall(line) + QUOTED.findall(line):
            if PLACEHOLDER.match(word):
                continue
            for token in word.split('_'):
                if token and len(token) <= 24:
                    tokens[token] += 1
    return [t for t, _ in tokens.most_common(window)]


def candidates(tokens, depth):
    """Every ordering of up to `depth` context tokens, joined the way the codebase joins them."""
    out = set(tokens)
    for n in range(2, depth + 1):
        for combo in itertools.permutations(tokens, n):
            out.add('_'.join(combo))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scripts', required=True)
    ap.add_argument('--known', nargs='+', required=True)
    ap.add_argument('--out')
    ap.add_argument('--window', type=int, default=22,
                    help='how many context tokens one hash may be built from (22). Cost is '
                         'window^depth per hash, and per hash the budget is enormous')
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--around', type=int, default=2, help='lines of context either side')
    ap.add_argument('--min-uses', type=int, default=2)
    args = ap.parse_args()

    known = read_known(args.known)

    # Gather every appearance first, so a hash used in six places is built from all six.
    where = collections.defaultdict(list)
    uses = collections.Counter()

    for base, _dirs, files in os.walk(args.scripts):
        for name in files:
            if not name.endswith(('.gsc', '.csc')):
                continue

            with open(os.path.join(base, name), encoding='utf-8', errors='replace') as fh:
                lines = fh.read().split('\n')

            heading = ''
            for number, line in enumerate(lines):
                match = FUNC.match(line)
                if match:
                    heading = match.group(1)

                for digest in HASHED.findall(line):
                    value = int(digest, 16)
                    if value in known:
                        continue
                    uses[value] += 1
                    if len(where[value]) < 40:
                        lo = max(0, number - args.around)
                        where[value].extend(lines[lo:number + args.around + 1])
                        if heading:
                            where[value].append(heading)

    targets = [h for h in where if uses[h] >= args.min_uses]
    print('unnamed hashes with %d or more uses: %d' % (args.min_uses, len(targets)))
    print('per hash: %d context tokens, depth %d, so at most %.3g candidates'
          % (args.window, args.depth, args.window ** args.depth))
    print('expected false matches per hash: %.2g' % (args.window ** args.depth / 2.0 ** 32))

    found = {}
    tried = 0
    for value in targets:
        tokens = context_tokens(where[value], args.window)
        if len(tokens) < 2:
            continue
        for candidate in candidates(tokens, args.depth):
            tried += 1
            if t7_hash(candidate) == value:
                found[value] = candidate
                break

    print('candidates tried: %.3g' % tried)
    print('names recovered : %d  (%.1f%% of the targets)'
          % (len(found), 100.0 * len(found) / max(1, len(targets))))

    if args.out:
        with open(args.out, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('hash,name\n')
            for value in sorted(found):
                fh.write('%08x,%s\n' % (value, found[value]))
        print('wrote %s' % args.out)


if __name__ == '__main__':
    main()
