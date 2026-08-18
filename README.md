# codehash

A GPU dictionary search for Call of Duty asset names.

From Black Ops 3 onward, Treyarch and Infinity Ward stopped shipping asset names. What a fast file
carries is a 60-bit hash, and an extractor that cannot resolve one has to fall back on writing
`ximage_a2309c9f2903.dds`. Community indexes cover the names somebody has already guessed; this
covers the rest, by composing candidate names out of a dictionary and hashing them until one lands
on a hash you are missing.

It does the same job as `acts hashbrutedictgpu`, which is the tool most people use. It exists
because that one is not bound by the hashing: it writes eight bytes per candidate into a 64 MB
buffer and copies the buffer back to be scanned on the CPU, which caps a run near 10^9 candidates
a second no matter how cheap the arithmetic gets. Removing that cap is most of what is here.

Measured on an RTX 3090: **1.28e10 candidates/s.** The command that produces that figure is in
[Throughput](#throughput), so you can check it on your own card rather than take the number.

## The hash

FNV-1a over the UTF-8 bytes, truncated to its low 60 bits:

```
h = 0xCBF29CE484222325
for each byte b:  h = (h XOR b) * 0x100000001B3   (mod 2^64)
hash = h AND 0x0FFFFFFFFFFFFFFF
```

Asset paths are hashed with forward slashes. A name written with backslashes hashes to something
else entirely, which is a mistake worth knowing about before it costs you a run.

## How it works

A candidate name is

```
<prefix> w1 _ w2 _ ... _ wD
```

split into a **stem** (everything up to and including the last separator) and a **leaf** (the final
word). One thread owns one stem: it hashes that stem once, then walks the entire dictionary as
leaves. Two things follow, and they are the point of the whole design.

- A candidate costs only its leaf - six or seven bytes - instead of rehashing thirty.
- Every thread in a warp is on the same leaf at the same time, so control flow does not diverge on
  word length and the dictionary read is one broadcast load per warp.

On top of that:

- **Nothing is written per candidate.** A hit does an `atomicAdd` into a small buffer; a miss
  writes nothing. That removes the per-candidate store, the PCIe copy and the host-side scan in
  one go - the three things that set the ceiling elsewhere.
- **No per-thread character buffer.** The running state is a single 64-bit register, so nothing
  spills to local memory and occupancy stays where it belongs.
- **Membership is a shared-memory bitmap**, sized so several blocks still fit on an SM. Only the
  few candidates that pass it touch the sorted table in global memory.
- **That table is pinned in L2** with `cudaAccessPropertyPersisting`, so the confirmations that do
  happen stay off DRAM. This is worth a factor of about 6.6 on its own, and it was found with
  Nsight Compute rather than guessed - two optimisations that looked obvious first (removing
  64-bit divisions, batching probes) were worth +12% and **-7%**.
- **The FNV prime is sparse.** `0x100000001B3 == 2^40 + 435`, so the 64x64 multiply that Ampere
  has to emulate becomes a shift plus a 64x32 multiply.

## Build

```
nvcc -O3 -arch=sm_86 -o codehash codehash.cu       # sm_86 = RTX 30xx, sm_89 = 40xx
```

No dependencies beyond the CUDA runtime. If the host has no `nvcc`, the devel image works and
leaves nothing behind:

```
docker run --rm --gpus all -v "$PWD":/w -w /w \
    nvidia/cuda:12.6.3-devel-ubuntu24.04 nvcc -O3 -arch=sm_86 -o codehash codehash.cu
```

## Use

```
codehash --targets <file> --dict <file> [--prefix S | --prefix-file F]
         [--depth N] [--depth-min N] [--mitm K] [--prefix-table]
         [--bitmap-bits N] [--chunk N] [--out F]
```

`--targets` is one hash per line, bare hex or `hash_<hex>`. `--dict` is one word per line.
**Write both with Unix line endings** - a trailing `\r` hashes into the candidate, matches
nothing, and reports nothing; it is the single most likely way to waste a run.

| option | what it does |
|---|---|
| `--dict a,b,c` | one file per position, in order. One file = flat search; several = positional. Fewer files than the depth and the last repeats, so `pos0,pos1,rest` is valid shorthand |
| `--prefix S` | fix a leading string. Every level not in the prefix comes out of the dictionary and multiplies the space by the vocabulary, so this is the cheapest lever you have |
| `--prefix-file F` | a list of prefixes, all run inside one process - a CUDA context costs ~200 ms to build, a quarter of the wall time of a fast pass |
| `--depth N` | how many dictionary words to compose. Cost is `prefixes * words^depth` |
| `--mitm K` | meet in the middle, forward over the first K words. The hash is invertible, so the two halves can be met in a table |
| `--prefix-table` | build a table of known prefixes and search backwards from the targets |
| `--bitmap-bits N` | membership bitmap size, as a power of two. 18 measured best - see [What does not help](#what-does-not-help) |
| `--chunk N` | stems per launch. Default 2^23 suits a headless box. **Drop to about 2^20 on Windows or any GPU driving a display**, where a kernel that outruns the driver's 2-second timeout is killed along with the process |

### A run, end to end

```
# 1. positional dictionaries - words ranked by how often they appear at THAT position.
#    --names takes CSV indexes of hash,name: the community databases, plus whatever
#    you have already recovered. Writes pos0.txt, pos1.txt, ..., last.txt, all.txt.
python tools/build_positional_dict.py --names known.csv --out-dir dict --positions 2

# 2. search
./codehash --targets targets.txt --dict dict/pos0.txt,dict/pos1.txt \
           --prefix ui_icon_inventory_zm_red_ --depth 2 --out found.txt

# 3. verify - never skip this
python tools/verify_names.py --targets targets.txt --names found.txt --slash --csv found.csv
```

## Verify everything

A search over a 60-bit space returns false positives by construction, about
`candidates * targets / 2^60` of them, and a hit is reported the moment a bitmap and a table
agree - which is exactly the agreement a truncated hash can fake. `tools/verify_names.py`
recomputes the hash of every name and checks it, costs nothing next to a ten hour run, and exits
non-zero if anything fails.

It also reports **collisions**: two different names landing on one 60-bit hash. Only one of them
is the asset, and no amount of hashing will tell you which. Those have to be settled by looking at
what the name points at.

### The CSV it writes

`--csv` writes every name the search proposed, whether it survived or not:

| column | meaning |
|---|---|
| `hash` | the 60-bit FNV-1a of the name in that row, as `hash_<hex>` - recomputed here, never copied from the search's own output, since that is the number being checked |
| `name` | the candidate string the search produced |
| `verified` | `yes` if that hash is in the target list, `no` if it is not |

`verified` says one thing precisely: the hash is in the list you passed to `--targets`. Check a
run against the targets it was aimed at and a `no` is a false positive the bitmap let through.
Check a wider set of names against a narrower target list - everything you have ever recovered
against what is still unresolved, say - and a `no` only means out of scope. The column does not
guess which of the two you meant, so the answer depends on the list you hand it.

Keeping the `no` rows rather than dropping them is deliberate. What a disappointing run proposed,
and how far off it was, is the most useful thing it produced, and two runs cannot be diffed if
each only lists its winners.

Rows sharing a `hash` with `verified` set to `yes` are the collisions above. Pick one by hand.

## Throughput

Measured on an RTX 3090, one prefix, a flat 50 000-word dictionary at depth 2:

```
$ ./codehash --targets targets.txt --dict dict_50000.txt --prefix i_c_t8_mp_spe_ --depth 2
depth 2: 2.5e+09 candidates, ~0.0 expected false hits
  0.2 s, 1.28e+10 candidates/s, 0 hit(s)
```

Two caveats worth stating plainly. That is the flat search, where every candidate costs one leaf
and nothing else. The prefix-table mode is far slower per unit of work - a real run of it managed
1.11e6 backward stems a second - because each stem probes a table that no longer fits in L2. And
the comparison against `acts hashbrutedictgpu`, which motivated writing this, is not re-measurable
here: `acts` is not installed on the machine these numbers come from, so no ratio is quoted. What
is verifiable by reading `acts` rather than running it is the architectural difference - it writes
eight bytes per candidate into a buffer and copies that buffer back to be scanned on the CPU, and
this does not.

## Progress and the ETA

Any run longer than one chunk prints where it is, how fast, and how much is left:

```
   3.4%  3.33e+05 stems/s  elapsed 25s  left 12m
```

The estimate is there because the cost of this search is `prefixes * words^depth`, so one more
dictionary position multiplies the whole run by the size of that position's vocabulary. The
difference between a setting that finishes overnight and one that finishes in fifty years is not
visible in the arguments, and without an estimate it is not visible until the end either. With
one it shows up in the first chunk, while changing your mind is still free.

The rate is averaged over the whole run rather than over the last chunk. Chunk timings wobble, and
a figure that jumps by a factor of two every second is one nobody trusts enough to act on.

## How long a run takes

Depth decides whether a run is possible at all, and the jump between depths is not a factor of two
or ten - it is the size of a vocabulary. Against 50 635 unresolved Black Ops 4 hashes with
positional vocabularies of 5 000 and 50 000 words, on one RTX 3090:

| what | work | time | |
|---|---|---|---|
| depth 2, prefix table over 3.0M prefixes | 4.05e10 stems | **10 h 09 min**, 4 470 names | timed |
| depth 3, same mode | 2.03e15 stems | **~58 years** | from that run's rate |
| depth 3, flat, one fixed prefix | 1.25e13 candidates | **12 min 47 s** | timed |

Only the middle row is arithmetic - the work divided by the rate the first row ran at. Nobody is
going to time fifty-eight years, and no precision on that figure changes what to do about it.

The flat run reaches 1.63e10 candidates/s, above the 1.28e10 of the depth 2 measurement above,
because a deeper search amortises the per-stem setup over more leaves. That 12 min 47 s was timed
before `--bitmap-bits` moved to 18, so it is if anything pessimistic now.

The second row is not a longer run, it is a different problem, and it is worth understanding why
the third row is so much cheaper. The prefix-table mode works backwards from the targets against a
table of three million known prefixes, so it can find a name whose prefix you could not have
guessed - but every stem probes a 100 MB table that does not fit in L2, and that probe sets the
pace. The flat search needs you to name the prefix, and in exchange never touches that table.

So when depth 3 is out of reach, the useful moves are not "wait longer":

- **Feed recovered names back into the dictionary.** A run that finds 4 470 names has also found
  the words in them, and those words were not in the vocabulary the run started with. Rebuilding
  the dictionary and repeating at depth 2 costs another ten hours and searches somewhere new.
- **Fix the prefix and go flat.** Twenty minutes per prefix, if you know which prefixes to attack.
- **Shorten the trailing vocabulary.** In the prefix-table mode, cutting the third position to 20
  words brings depth 3 down to about 8 days; 50 words is 21 days; 500 is 212 days. Rarely worth it.

If you see `EXCEEDS L2` in the header, move the split or shorten the leading positional lists
before you wait ten hours to find out what it cost.

## When the hash stops being the constraint

Everything above searches names as combinations of dictionary words, and that is the right shape
only while a name is short. Measured on 22 481 recovered Black Ops 4 names, the median name has
**nine** words and only 4.4% have three or fewer. With a 50 000 word vocabulary the space passes
2^60 at 3.8 words - so for the overwhelming majority of real names there are more word sequences
than there are hashes, roughly 2^80 of them per target, every one an equally valid preimage.

That is worth stating plainly because it decides where effort goes. Past four words the hash is a
checksum, not a filter: no arithmetic can pick the true name out of 2^80 arithmetically identical
ones. Only a prior can. Making the search faster cannot help, and the measurements in the section
above show there was not much speed left to find anyway.

`mitm_frag.cu` is the answer to that. It searches **fragments** instead of words - every prefix and
every suffix cut at a separator out of the names already known - and joins them by meeting in the
middle on the 60-bit state, which FNV-1a permits because it is invertible. A missing name that
recombines known parts falls out immediately.

Two properties make it work:

- **It is precise by construction.** 883 488 prefixes against 3 107 168 suffixes over 50 635
  targets is 2.7e12 pairs, and 2.7e12 pairs is 0.12 expected false matches for the entire run.
  Compare a word search, where the space is so much larger than 2^60 that most matches are noise.
- **The two sides cost differently.** A prefix costs sixteen bytes of table; a suffix costs a
  backward walk against every target. Reach is bought on the prefix side.

Results on the same 50 635 unresolved hashes, each round feeding its finds back in as corpus:

| | names | share of targets | time |
|---|---|---|---|
| word search, depth 2, prefix table | 4 470 | 8.8% | 10 h 09 min |
| fragments, recombination only | 641 | 1.3% | 19 s |
| + generic 256-word extension | 1 857 | 3.7% | 35 s |
| + per-prefix bigram continuations | 5 748 | 11.3% | 32 s |
| + four feedback rounds | 9 478 | 18.6% | 32 s each |
| + continuation budget by prefix frequency | **13 293** | **26.1%** | 32 s each |

The last row is worth its own sentence. Spreading the continuation budget evenly over prefixes
treats a word offered to `mc/` as worth the same as one offered to
`i_c_t8_mp_spe_outrider_apocalypse_`, and it is not: the short one heads a great many names. Giving
the commonest twenty thousand prefixes a deeper list costs nothing measurable and found 3 800 more
names. Directory prefixes make the case on their own - 275 of them in the whole corpus, heading 3%
of known names and **21%** of the names this search recovers.

The jump from a generic extension to per-prefix continuations is the whole argument in one line:
2.4x the names for less than half the search, because offering `i_c_t8_mp_spe_` the words that
have actually followed `spe` beats offering it the 256 commonest words in the game.

Everything reported here was recomputed and checked with `tools/verify_names.py`. That is not
ceremony: at a continuation cap of 512 the slot index overflowed eight bits into the prefix index,
and 355 of 8 125 reported names came back as the wrong string. They still matched a target state,
so nothing but recomputing the hash could have caught it.

### Running it

```
python tools/build_fragments.py known.csv found_so_far.txt --out-dir w --cap 512
./mitm_frag --prefixes w/prefixes.txt --suffixes w/suffixes.txt --targets targets.txt             --extend w/words.txt --cont-offset w/cont_offset.bin --cont-index w/cont_index.bin             --out found.txt
python tools/verify_names.py --targets targets.txt --names found.txt --slash --csv found.csv
```

Then put `found.txt` back on the `build_fragments.py` command line and run it again. The rounds
compound and then saturate - 5 748, 7 966, 8 963, 9 413, 9 478 - and when the increment falls off
the corpus has given what it has.

### One more free factor of sixteen

The stored hash is truncated to 60 bits, so the top four bits of the state are gone, and the
obvious reading is that any backward pass must try all sixteen. `codehash` does exactly that. It
does not have to: multiplication mod 2^64 truncated to 60 bits equals multiplication mod 2^60 of
the truncated operands, and a byte XOR cannot reach bit 60, so the whole chain is closed under
truncation. `mitm_frag` runs entirely mod 2^60 and skips the sixteen guesses. Verified against the
full 64-bit computation over three thousand random splits.

## What does not help

Measured on the 3090 rather than reasoned about, because the reasoning was wrong twice.

**A bigger membership bitmap.** With 50 635 targets in 2^17 bits the false positive rate is 38.6%,
which sounds ruinous - more than a third of candidates go on to probe the table. Doubling the
bitmap halves the rate and gains 5%; doubling it again cuts the rate to 9.7% and costs a factor of
two, because 64 KB of shared memory leaves one block per SM and occupancy collapses from 83% to
16%. 2^18 is the sweet spot and it is worth 5%, not 2x.

| `--bitmap-bits` | false positives | occupancy | rate |
|---|---|---|---|
| 17 | 38.6% | 83% | 1.66e10/s |
| 18 | 19.3% | 50% | **1.75e10/s** |
| 19 | 9.7% | 16% | 7.72e9/s |

**Making the hash itself cheaper.** Three dictionaries of 5 000 words each, identical in every way
but word length:

| mean word length | rate |
|---|---|
| 3.5 | 1.67e10/s |
| 7.5 | 1.64e10/s |
| 15.5 | 1.55e10/s |

4.4x the characters costs 7%. Solving for the two terms, a character is worth 0.66% of the fixed
per-candidate cost, so at a realistic 7.5 characters **the hashing is about 5% of the runtime**.
Whatever is setting the pace, it is not the arithmetic - it is the per-leaf dictionary load, the
bitmap probe and the loop around them.

That measurement retires an optimisation that looked excellent on paper. FNV-1a decomposes: for a
word `w` of length `L`,

```
Hash(h, w) = h * P^L + F(h mod 128, w)
```

The word's whole contribution depends on the running state through **seven bits** and nothing else
- verified exhaustively over the low byte and across word lengths from 1 to 31. So a table of
128 rows by dictionary size turns a leaf of any length into one multiply, one lookup and one add,
and the multiply hoists out if the dictionary is sorted by length. It is a real property and it is
the reason FNV should never be used where preimages matter. As a speed optimisation here it is
chasing 5%, and it would cost bucketing every stem by its residue class to keep a warp reading one
row. Not worth it. Someone attacking the fixed 95% - a smaller dictionary entry than 32 bytes, a
cheaper probe - would be aiming at the right thing.

## Files

```
codehash.cu                       word-combination searcher, for short names
mitm_frag.cu                      fragment meet in the middle, for everything else
tools/build_fragments.py          cut a corpus into prefixes, suffixes and continuations
tools/build_positional_dict.py    per-position dictionaries from known names
tools/verify_names.py             recompute and check, plus collision reporting
```

## Credit

The community hash indexes - [ate47/HashIndex](https://github.com/ate47/HashIndex) and the
databases published around `acts` - are where most known Call of Duty names come from, and where
any dictionary worth searching starts. This tool is for what is left after those.
