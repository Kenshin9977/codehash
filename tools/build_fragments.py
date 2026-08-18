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
    ap.add_argument('--deep-top', type=int, default=0, metavar='N',
                    help='give the N commonest prefixes a bigger continuation list than the rest')
    ap.add_argument('--deep-cap', type=int, default=4096)
    ap.add_argument('--bigram-cap', type=int, default=0, metavar='C',
                    help='additionally offer each prefix its C likeliest two-word continuations. '
                         'Nothing requires a continuation to be one word, so this reaches names '
                         'with a wholly new two-word middle at the same cost per entry')
    ap.add_argument('--full-vocab-top', type=int, default=0, metavar='N',
                    help='give the N commonest prefixes every word in the vocabulary. Meant for '
                         'directory prefixes: there are only 275 of them, so a complete sweep of '
                         'that shape is affordable where it would not be for anything longer')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    names = load(args.corpus)
    print('corpus: %d names' % len(names))

    prefix = collections.Counter()
    suffix = collections.Counter()
    follow = collections.defaultdict(collections.Counter)
    follow2 = collections.defaultdict(collections.Counter)
    vocabulary = collections.Counter()

    for name in names:
        for i, ch in enumerate(name):
            if ch in '_/':
                prefix[name[:i + 1]] += 1
                suffix[name[i + 1:]] += 1

        for token in name.replace('/', '_').split('_'):
            if token:
                vocabulary[token] += 1

        tokens = [t for t in name.replace('/', '_').split('_') if t]
        for a, b in zip(tokens, tokens[1:]):
            follow[a][b] += 1
        for a, b, c in zip(tokens, tokens[1:], tokens[2:]):
            follow2[a][b + '_' + c] += 1

    # Ordered by frequency throughout, which is the whole prior: where two fragments collide on a
    # state the commoner one wins, and a truncated list keeps the likelier half.
    prefixes = [f for f, _ in prefix.most_common() if 0 < len(f) <= args.max_fragment]
    suffixes = [f for f, _ in suffix.most_common() if 0 < len(f) <= args.max_fragment]

    # Budget by probability mass, not evenly.
    #
    # A short prefix stands in front of a great many names, so a word offered to it is worth far
    # more than the same word offered to something long and specific. Directory prefixes make the
    # case: there are 275 of them in the whole corpus, they head only 3% of known names - and 21%
    # of the names this search recovers. Giving that handful a deep list costs nothing measurable
    # and reaches the shape they are over-represented in.
    vocab = {}
    offsets = [0]
    index = []
    common = [w for w, _ in vocabulary.most_common()]

    for rank, p in enumerate(prefixes):
        cap = args.deep_cap if rank < args.deep_top else args.cap
        tokens = [t for t in p.rstrip('_/').replace('/', '_').split('_') if t]
        last = tokens[-1] if tokens else ''

        offered = [w for w, _ in follow.get(last, collections.Counter()).most_common(cap)]
        if rank < args.full_vocab_top:
            offered = common
        if args.bigram_cap:
            offered = offered + [w for w, _ in
                                 follow2.get(last, collections.Counter()).most_common(args.bigram_cap)]

        for word in offered:
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
