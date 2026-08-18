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

Measured on an RTX 3090: **1.08e10 candidates/s.** The command that produces that figure is in
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
  0.2 s, 1.08e+10 candidates/s, 0 hit(s)
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
| depth 3, flat, one fixed prefix | 1.25e13 candidates | **~20 min** | from the flat rate above |

Only the first row is a stopwatch reading. The other two divide the work by a rate measured on
this card, which is the right way round: the point of the table is that the difference between
them is six orders of magnitude, and no amount of precision on the second row changes what to do
about it.

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

## Files

```
codehash.cu                       the searcher
tools/build_positional_dict.py    per-position dictionaries from known names
tools/verify_names.py             recompute and check, plus collision reporting
```

## Credit

The community hash indexes - [ate47/HashIndex](https://github.com/ate47/HashIndex) and the
databases published around `acts` - are where most known Call of Duty names come from, and where
any dictionary worth searching starts. This tool is for what is left after those.
