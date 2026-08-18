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
    ap.add_argument('--prior', nargs='*', default=[], metavar='FILE',
                    help='names used only to learn vocabulary and continuations, never to supply '
                         'a prefix or a suffix. This is how a database far larger than the search '
                         'can afford still contributes: the community archive holds three million '
                         'names across every title, whose fragment lists would need a 34 GB table '
                         'and carry 394 expected false matches, but whose word statistics cost '
                         'nothing and nearly double the vocabulary')
    ap.add_argument('--boost', nargs='*', default=[], metavar='FILE',
                    help='names whose families should be favoured when the deep budget is handed '
                         'out. Pass the previous round output here: the families that just paid '
                         'are the ones with something left to give')
    ap.add_argument('--boost-weight', type=int, default=200)
    ap.add_argument('--full-vocab-top', type=int, default=0, metavar='N',
                    help='give the N commonest prefixes every word in the vocabulary. Meant for '
                         'directory prefixes: there are only 275 of them, so a complete sweep of '
                         'that shape is affordable where it would not be for anything longer')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    names = load(args.corpus)
    print('corpus: %d names' % len(names))

    # Rank prefixes by where names are still being found, not by where they already are.
    #
    # Ranking on corpus frequency hands the deepest continuation lists to the commonest prefixes,
    # which is precisely the part of the space that is already exhausted. The measurement is not
    # subtle: i_mtl_ has 85 164 known names and yields 1.3 new ones per known one, while au_wz_
    # had ZERO known names and yielded 549. Families the community never covered are where the
    # unresolved hashes are, which is survivorship bias stated as a search strategy - what you
    # have already found tells you where you have already looked, not where to look next.
    #
    # So a name from the last round counts for two hundred, and its family climbs.
    boosted = load(args.boost) if args.boost else set()
    if boosted:
        print('boost: %d names weighted x%d' % (len(boosted), args.boost_weight))

    prior = load(args.prior) if args.prior else set()
    if prior:
        print('prior: %d names, for continuations only' % len(prior))

    def learn(name):
        tokens = [t for t in name.replace('/', '_').split('_') if t]
        for token in tokens:
            vocabulary[token] += 1
        for a, b in zip(tokens, tokens[1:]):
            follow[a][b] += 1
        for a, b, c in zip(tokens, tokens[1:], tokens[2:]):
            follow2[a][b + '_' + c] += 1

    prefix = collections.Counter()
    suffix = collections.Counter()
    follow = collections.defaultdict(collections.Counter)
    follow2 = collections.defaultdict(collections.Counter)
    vocabulary = collections.Counter()

    for name in names:
        weight = args.boost_weight if name in boosted else 1
        for i, ch in enumerate(name):
            if ch in '_/':
                prefix[name[:i + 1]] += weight
                suffix[name[i + 1:]] += 1


        learn(name)

    # The prior names teach the model and nothing else - no prefix, no suffix, no entry in the
    # table. They cost a pass over the text and buy a much better answer to "what could follow
    # this word".
    for name in prior:
        learn(name)

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
