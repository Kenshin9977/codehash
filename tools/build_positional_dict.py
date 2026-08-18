#!/usr/bin/env python3
"""Build one dictionary per word position, ranked by how often each word appears THERE.

A flat dictionary treats a name as an independent draw from one vocabulary at every position,
which is not how these names are built. After the map token, the first word comes from a small
set of categories - shield, tool, hand, fuse, dynamite, eqp - while the last is usually a state
or a colour: pap, charged, uncharged, completed, yellow. Searching V^d when the real space is
V1*V2*...*Vd wastes most of the run on combinations the game would never produce.

Two files come out of this, and they answer the two halves of the problem:

  pos<N>.txt   the vocabulary for position N, most frequent first
  last.txt     the vocabulary for the FINAL position, which is distributed differently

Ranking each list by frequency also gives probability-ordered enumeration for free, without a
priority queue on the GPU: truncate every list to its top K and the search covers the most
likely names first. Run K=256, then 1024, then 4096 - the cost of the small runs is negligible
next to the large one, and a name found at K=256 is found in milliseconds instead of an hour.

    python scripts/build_positional_dict.py --names index/fnv1a_ximages.csv \
        --family ui_icon_inventory_ --out-dir work/pos --positions 4
"""

import argparse
import collections
import csv
import os
import re

TOKEN = re.compile(r'^[a-z0-9]+$')
HEXISH = re.compile(r'^[0-9a-f]{6,}$')


def is_word(t):
    return 2 <= len(t) <= 15 and TOKEN.match(t) and not t.isdigit() and not HEXISH.match(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--names', nargs='+', required=True, help='CSV indexes of hash,name')
    ap.add_argument('--family', default='', help='only names starting with this, e.g. ui_icon_')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--positions', type=int, default=4)
    ap.add_argument('--max-words', type=int, default=20000)
    ap.add_argument('--min-count', type=int, default=2)
    args = ap.parse_args()

    per_pos = [collections.Counter() for _ in range(args.positions)]
    last = collections.Counter()
    overall = collections.Counter()
    kept = 0

    for path in args.names:
        with open(path, encoding='utf-8', errors='replace') as fh:
            for row in csv.reader(fh):
                if len(row) < 2:
                    continue
                name = row[1].strip().lower()
                if args.family and not name.startswith(args.family):
                    continue
                tail = name[len(args.family):].split('_')
                tail = [t for t in tail if is_word(t)]
                if not tail:
                    continue
                kept += 1
                overall.update(tail)
                last[tail[-1]] += 1
                for i, t in enumerate(tail[:args.positions]):
                    per_pos[i][t] += 1

    print('%d names kept in family "%s"' % (kept, args.family or '(all)'))
    if not kept:
        return

    os.makedirs(args.out_dir, exist_ok=True)

    def write(counter, path, label):
        # A position seen in only a handful of names has no usable statistics of its own, so it
        # falls back to the overall frequency rather than to a list of three words that happens
        # to describe the few examples we have.
        source = counter if sum(counter.values()) >= 200 else overall
        note = '' if source is counter else '  (too little data, fell back to overall frequency)'
        words = [w for w, c in source.most_common() if c >= args.min_count][:args.max_words]
        with open(path, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('\n'.join(words) + '\n')
        print('  %-10s %6d words -> %s%s' % (label, len(words), path, note))
        return len(words)

    sizes = []
    for i in range(args.positions):
        sizes.append(write(per_pos[i], os.path.join(args.out_dir, 'pos%d.txt' % i), 'position %d' % i))
    write(last, os.path.join(args.out_dir, 'last.txt'), 'last')
    write(overall, os.path.join(args.out_dir, 'all.txt'), 'overall')

    print()
    flat = len(overall)
    for d in range(2, args.positions + 1):
        positional = 1
        for s in sizes[:d]:
            positional *= s
        print('  depth %d: %.3g positional candidates against %.3g flat  (/%.0f)'
              % (d, positional, flat ** d, (flat ** d) / max(positional, 1)))

    print()
    print('  enumeration par probabilite : tronquer chaque liste a K croissant')
    for k in (256, 1024, 4096):
        print('    K=%-5d depth 3: %.3g candidates' % (k, min(k, flat) ** 3))


if __name__ == '__main__':
    main()
