#!/usr/bin/env python3
"""Cut a corpus of known names into the fragment lists and continuations mitm_frag searches.

    python tools/build_fragments.py --corpus known.txt [found1.txt found2.txt ...] --out-dir work

Three things come out of here, and the third is what makes the search worth running.

  prefixes    every prefix of every name, cut at a separator and keeping the separator, ordered by
              how often it occurs. Cheap to add to: a prefix costs sixteen bytes of table.
  suffixes    the same from the other end. Expensive to add to: every suffix is walked backwards
              against every target.
  continuations  per prefix, the words that have actually followed its last token somewhere in the
              corpus. This is the prior, and it is where the yield comes from - offering each
              prefix its own 109 plausible words instead of the same 256 common ones found 2.4x
              more names for less than half the search.

Feed the names a run finds back in as extra corpus and repeat. The fragments of a recovered name
unlock its neighbours, so the rounds compound: 5748 names after one round became 7966 after two.
"""

import argparse
import collections
import os
import struct


def load(paths):
    names = set()
    for path in paths:
        if not os.path.exists(path):
            print('  missing: %s' % path)
            continue
        before = len(names)
        with open(path, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                line = line.rstrip('\n').strip()
                if not line or line.startswith('#'):
                    continue
                # Accept a bare name or a hash,name CSV row, so a verifier's output feeds straight
                # back in without a conversion step.
                if ',' in line and line.split(',', 1)[0].startswith('hash_'):
                    line = line.split(',')[1]
                name = line.replace(chr(92), '/')
                if len(name) > 2 and not name.startswith('hash_'):
                    names.add(name)
        print('  %-40s +%d' % (os.path.basename(path), len(names) - before))
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus', nargs='+', help='name lists, plain or hash,name CSV')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--cap', type=int, default=512,
                    help='most continuations to offer one prefix (512). The slot index is stored '
                         'in sixteen bits, so anything up to 65535 is safe')
    ap.add_argument('--max-fragment', type=int, default=120)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    names = load(args.corpus)
    print('corpus: %d names' % len(names))

    prefix = collections.Counter()
    suffix = collections.Counter()
    follow = collections.defaultdict(collections.Counter)

    for name in names:
        for i, ch in enumerate(name):
            if ch in '_/':
                prefix[name[:i + 1]] += 1
                suffix[name[i + 1:]] += 1

        tokens = [t for t in name.replace('/', '_').split('_') if t]
        for a, b in zip(tokens, tokens[1:]):
            follow[a][b] += 1

    # Ordered by frequency throughout, which is the whole prior: where two fragments collide on a
    # state the commoner one wins, and a truncated list keeps the likelier half.
    prefixes = [f for f, _ in prefix.most_common() if 0 < len(f) <= args.max_fragment]
    suffixes = [f for f, _ in suffix.most_common() if 0 < len(f) <= args.max_fragment]

    vocab = {}
    offsets = [0]
    index = []
    for p in prefixes:
        tokens = [t for t in p.rstrip('_/').replace('/', '_').split('_') if t]
        last = tokens[-1] if tokens else ''
        for word, _ in follow.get(last, collections.Counter()).most_common(args.cap):
            if word not in vocab:
                vocab[word] = len(vocab)
            index.append(vocab[word])
        offsets.append(len(index))

    entries = len(prefixes) + len(index)
    pairs = entries * len(suffixes)
    print('prefixes %d, suffixes %d' % (len(prefixes), len(suffixes)))
    print('continuations %.1f M (%.0f per prefix), entries %.1f M'
          % (len(index) / 1e6, len(index) / max(1, len(prefixes)), entries / 1e6))
    print('pairs %.2e' % pairs)

    slots = 1
    while slots < entries * 2:
        slots *= 2
    print('table %d slots = %.1f GB' % (slots, slots * 16 / 1e9))

    words = [None] * len(vocab)
    for word, i in vocab.items():
        words[i] = word

    def write(name, lines):
        with open(os.path.join(args.out_dir, name), 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('\n'.join(lines) + '\n')

    write('prefixes.txt', prefixes)
    write('suffixes.txt', suffixes)
    write('words.txt', words)
    open(os.path.join(args.out_dir, 'cont_offset.bin'), 'wb').write(
        struct.pack('<%dI' % len(offsets), *offsets))
    open(os.path.join(args.out_dir, 'cont_index.bin'), 'wb').write(
        struct.pack('<%dI' % len(index), *index))
    print('written to %s' % args.out_dir)


if __name__ == '__main__':
    main()
