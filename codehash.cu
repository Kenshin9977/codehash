// codehash - dictionary search for Call of Duty asset names, on one GPU.
//
// Same job as `acts hashbrutedictgpu`: compose dictionary words into a name, hash it with the
// game's FNV-1a, and report the ones that land on a hash we could not resolve. Written separately
// because acts' kernel is bounded by things that have nothing to do with hashing - it writes eight
// bytes per candidate into a 64 MB buffer and copies that buffer back to the host to be scanned,
// which caps a run near 1e9 candidates a second whatever the arithmetic costs.
//
// Build:   nvcc -O3 -arch=sm_86 -o codehash codehash.cu        (sm_86 = RTX 30xx; 89 = 40xx)
// Run:     codehash --targets icons.txt --dict dict_5000.txt \
//                   --prefix ui_icon_inventory_zm_red_ --depth 3
//
// The structure is what makes the optimisations possible, so it comes first. A name is
//
//     <prefix> w1 _ w2 _ ... _ wD
//
// and it is split into a STEM (everything up to and including the last separator) and a LEAF (the
// final word). One thread owns one stem; it hashes that stem once, then walks the whole dictionary
// as leaves. Two consequences, and they are the whole point:
//
//   - a candidate costs only its leaf, six or seven bytes, instead of rehashing thirty;
//   - every thread is on the same leaf at the same time, so control flow is uniform, there is no
//     divergence on word length, and the dictionary read is one broadcast load per warp.
//
// On top of that:
//
//   - nothing is written per candidate. A hit does an atomicAdd into a small buffer; a miss writes
//     nothing at all. That removes the memory traffic, the PCIe copy and the host-side scan.
//   - no per-thread character buffer. State is one 64-bit register, so nothing spills to local
//     memory and occupancy stays where it should be.
//   - membership is a bitmap in shared memory, sized so several blocks still fit per SM. Only the
//     few candidates that pass it touch the sorted table in global memory.
//   - that sorted table is pinned in L2 (`cudaAccessPropertyPersisting`), so the confirmations
//     that do happen stay off DRAM.
//   - the FNV prime is sparse: 0x100000001B3 == 2^40 + 435, so the 64x64 multiply that Ampere has
//     to emulate becomes a shift plus a 64x32 multiply.

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define FNV_OFFSET 0xCBF29CE484222325ULL
#define MASK60     0x0FFFFFFFFFFFFFFFULL

// One dictionary entry is 32 bytes: up to 31 characters, then the length. Fixed stride so an entry
// is two aligned 16-byte loads at an address every thread in the block shares.
//
// It was 16 bytes, which was fine while entries were single words. It is not fine for the tails a
// name ends with - `foliage_rose_bush` is eighteen characters - and a dropped entry is a name the
// search can never reach, so the width is set by the longest thing worth searching for, not by the
// narrowest load.
#define WORD_STRIDE 32
#define WORD_MAX    31

struct Word { uint4 a, b; };

#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        cudaError_t err_ = (call);                                                            \
        if (err_ != cudaSuccess) {                                                            \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            std::exit(1);                                                                     \
        }                                                                                     \
    } while (0)

// ---------------------------------------------------------------------------------------------
// The hash

__host__ __device__ __forceinline__ uint64_t FnvStep(uint64_t h, uint32_t c)
{
    h ^= (uint64_t)c;
    // 0x100000001B3 == (1 << 40) + 435. Ampere has no 64-bit integer multiply; the general form
    // costs an IMUL plus two IMADs, about sixteen cycles. Splitting on the sparsity leaves a
    // funnel shift and a 64x32 multiply. The identity is exact: multiplication distributes, and
    // both sides wrap the same way modulo 2^64.
    return (h << 40) + h * 435ULL;
}

__host__ inline uint64_t FnvString(uint64_t h, const char* s, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c >= 'A' && c <= 'Z') c += 32;   // the game lowercases
        else if (c == '\\')       c = '/';   // and normalises the separator
        h = FnvStep(h, c);
    }
    return h;
}

// ---------------------------------------------------------------------------------------------
// The kernel

// Folds one dictionary word into the hash, straight out of the registers holding it.
//
// The bytes are never addressed. Taking `&w` on a local uint4 would put it in local memory, which
// lives in DRAM, and the whole reason this kernel exists is that state stays in registers - so the
// loop is fully unrolled and the component is selected at compile time. `n` is uniform across the
// block, so the early exit costs no divergence.
__device__ __forceinline__ uint64_t HashWord(uint64_t h, Word w, uint32_t n)
{
    uint32_t c = w.a.x;
#pragma unroll
    for (uint32_t i = 0; i < WORD_MAX; i++) {
        if (i == 4)  c = w.a.y;
        if (i == 8)  c = w.a.z;
        if (i == 12) c = w.a.w;
        if (i == 16) c = w.b.x;
        if (i == 20) c = w.b.y;
        if (i == 24) c = w.b.z;
        if (i == 28) c = w.b.w;
        if (i >= n) break;
        h = FnvStep(h, c & 0xFF);
        c >>= 8;
    }
    return h;
}

// The same fold, on the low 32 bits only.
//
// FNV-1a truncated to k bits is exactly the FNV-1a chain modulo 2^k with the multiplier itself
// truncated - the carries only ever propagate upward, so the low half never learns about the high
// half. Verified over twenty thousand random strings at k = 8, 16 and 32. Truncated to 32 bits the
// multiplier 0x100000001B3 becomes 435, so the step is one native 32-bit multiply-add instead of
// the shift, 64x32 multiply and 64-bit add that the full width costs.
//
// The leaf sweep runs on this cheap chain alone. The bitmap only ever looks at the low bits, which
// the two chains share, so it is exactly as selective; the full 64-bit hash is recomputed for the
// one candidate in a thousand that gets through.
__device__ __forceinline__ uint32_t HashWord32(uint32_t h, Word w, uint32_t n)
{
    uint32_t c = w.a.x;
#pragma unroll
    for (uint32_t i = 0; i < WORD_MAX; i++) {
        if (i == 4)  c = w.a.y;
        if (i == 8)  c = w.a.z;
        if (i == 12) c = w.a.w;
        if (i == 16) c = w.b.x;
        if (i == 20) c = w.b.y;
        if (i == 24) c = w.b.z;
        if (i == 28) c = w.b.w;
        if (i >= n) break;
        h = (h ^ (c & 0xFF)) * 435u;
        c >>= 8;
    }
    return h;
}

// ---------------------------------------------------------------------------------------------
// Walking the hash backwards
//
// The FNV prime is odd, so it has a modular inverse and every step is reversible:
//
//     forward   h' = (h ^ c) * P
//     backward  h  = (h' * P^-1) ^ c        P^-1 = 0xCE965057AFF6957B
//
// which makes a meet-in-the-middle possible. Hash the first half of the words forward from the
// prefix and keep the states; walk the second half backward from each target hash; a name exists
// wherever the two meet. Instead of V^D candidates the cost becomes V^k forward plus 16*T*V^(D-k)
// backward - at depth four with five thousand words and a hundred and fifty targets, ten thousand
// times fewer operations.
//
// The sixteen is the price of the 60-bit mask: the top four bits of the real hash were thrown away
// when the pool was read, so every target is sixteen possible 64-bit values.
#define FNV_INV 0xCE965057AFF6957BULL

__host__ __device__ __forceinline__ uint64_t FnvUnstep(uint64_t h, uint32_t c)
{
    return (h * FNV_INV) ^ (uint64_t)c;
}

// Peels one word off the end of the hash. The characters come off in reverse, so the unrolled
// selection runs from the last byte down.
__device__ __forceinline__ uint64_t UnhashWord(uint64_t h, Word w, uint32_t n)
{
#pragma unroll
    for (int i = WORD_MAX - 1; i >= 0; i--) {
        if ((uint32_t)i >= n) continue;
        uint32_t chunk = (i < 16)
            ? (i < 8  ? (i < 4  ? w.a.x : w.a.y) : (i < 12 ? w.a.z : w.a.w))
            : (i < 24 ? (i < 20 ? w.b.x : w.b.y) : (i < 28 ? w.b.z : w.b.w));
        h = FnvUnstep(h, (chunk >> ((i & 3) * 8)) & 0xFF);
    }
    return h;
}

// Per-position vocabularies. A name is not an independent draw from one word list at every
// position: after `ui_icon_` only 78 distinct words are ever seen, against 2991 overall. Giving
// each position its own list turns V^d into V0*V1*...*Vd, which at depth 4 is twenty-two thousand
// times smaller - the difference between an hour and a fifth of a second.
#define MAX_DEPTH 8
struct PosDict {
    uint32_t base[MAX_DEPTH];   // first entry of this position's list, in the packed array
    uint32_t count[MAX_DEPTH];  // how many words it holds
};

__device__ __forceinline__ bool BitmapHas(const uint32_t* bitmap, uint32_t mask, uint64_t h)
{
    uint32_t bit = (uint32_t)(h & mask);
    return (bitmap[bit >> 5] >> (bit & 31)) & 1u;
}

__device__ __forceinline__ bool TableHas(const uint64_t* __restrict__ table, uint32_t n, uint64_t h)
{
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = (lo + hi) >> 1;
        uint64_t v = __ldg(&table[mid]);
        if (v < h)      lo = mid + 1;
        else if (v > h) hi = mid;
        else            return true;
    }
    return false;
}

// Each thread owns one stem and sweeps every leaf. `stemDepth` is the number of words before the
// leaf, so depth 1 means stemDepth 0 and the prefix state is used as is - which is why the prefix
// is expected to end with its own separator.
__global__ __launch_bounds__(256)
void Search(const Word* __restrict__ dict,
            PosDict      pd,
            uint64_t     prefixState,
            uint32_t     stemDepth,
            uint64_t     stemBegin,
            uint64_t     stemEnd,
            const uint32_t* __restrict__ bitmapSrc,
            uint32_t     bitmapWords,
            uint32_t     bitmapMask,
            const uint64_t* __restrict__ table,
            uint32_t     tableCount,
            uint64_t* __restrict__ hits,
            uint32_t* __restrict__ hitCount,
            uint32_t     hitMax)
{
    extern __shared__ uint32_t bitmap[];

    // Loaded once per block and reused for the block's whole lifetime, which is why the grid is
    // sized to the machine rather than to the work: a persistent block amortises this away.
    for (uint32_t i = threadIdx.x; i < bitmapWords; i += blockDim.x) {
        bitmap[i] = bitmapSrc[i];
    }
    __syncthreads();

    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

    for (uint64_t stem = stemBegin + blockIdx.x * blockDim.x + threadIdx.x;
         stem < stemEnd;
         stem += stride) {

        // Decode the stem into word indices and hash it, once for the whole dictionary sweep that
        // follows. The 64-bit divisions here are expensive and deliberately are: there are at most
        // three of them per several thousand candidates.
        uint64_t h0 = prefixState;
        uint64_t x = stem;
        for (uint32_t k = 0; k < stemDepth; k++) {
            uint32_t n = pd.count[k];
            uint32_t idx = (uint32_t)(x % n);   // mixed radix: each position has its own size
            x /= n;

            Word w = dict[pd.base[k] + idx];
            h0 = HashWord(h0, w, (w.b.w >> 24) & 0xFF);
            h0 = FnvStep(h0, '_');
        }
        const uint32_t h0lo = (uint32_t)h0;

        // The leaf sweep. `j` is the same for every thread in the block, so the load below is a
        // broadcast, the trip count of the inner loop is uniform, and no warp diverges. Only the
        // low 32 bits are carried here; the full width is recomputed for the rare survivor.
        const uint32_t leafBase = pd.base[stemDepth];
        const uint32_t leaves = pd.count[stemDepth];
        for (uint32_t j = 0; j < leaves; j++) {
            Word w = dict[leafBase + j];
            uint32_t len = (w.b.w >> 24) & 0xFF;
            uint32_t h32 = HashWord32(h0lo, w, len);

            // Almost every candidate stops on this line, in shared memory, and writes nothing.
            if (BitmapHas(bitmap, bitmapMask, (uint64_t)h32)) {
                uint64_t h = HashWord(h0, w, len) & MASK60;
                if (TableHas(table, tableCount, h)) {
                    uint32_t slot = atomicAdd(hitCount, 1u);
                    if (slot < hitMax) hits[slot] = stem * leaves + j;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Meet in the middle

// Open addressing, linear probing, keyed on the 64-bit state. A state of all ones marks a free
// slot; the real chain never produces it in practice and a lost collision would only cost one
// candidate out of tens of millions.
#define TABLE_EMPTY 0xFFFFFFFFFFFFFFFFULL

__global__ __launch_bounds__(256)
void MitmForward(const Word* __restrict__ dict, PosDict pd, uint64_t prefixState,
                 uint32_t splitAt, uint64_t count,
                 uint64_t* __restrict__ keys, uint32_t* __restrict__ vals, uint32_t tableMask)
{
    for (uint64_t idx = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
         idx < count;
         idx += (uint64_t)gridDim.x * blockDim.x) {

        uint64_t h = prefixState;
        uint64_t x = idx;
        for (uint32_t k = 0; k < splitAt; k++) {
            uint32_t n = pd.count[k];
            Word w = dict[pd.base[k] + (uint32_t)(x % n)];
            x /= n;
            h = HashWord(h, w, (w.b.w >> 24) & 0xFF);
            h = FnvStep(h, '_');
        }

        uint32_t slot = (uint32_t)(h ^ (h >> 32)) & tableMask;
        for (;;) {
            uint64_t prev = atomicCAS((unsigned long long*)&keys[slot], TABLE_EMPTY, h);
            if (prev == TABLE_EMPTY) { vals[slot] = (uint32_t)idx; break; }
            if (prev == h) break;                       // same state reached twice, keep the first
            slot = (slot + 1) & tableMask;
        }
    }
}

// The backward half, mirrored on the forward kernel's shape: a thread owns one BACKWARD STEM -
// a target, a top nibble, and the words from the split to the end - peels all of that once, then
// sweeps the word next to the split as its leaf.
//
// The first version decoded every candidate's whole tuple from its index, which meant several
// 64-bit divisions per candidate. Those are emulated on the GPU and cost more than the hashing
// they were feeding. Here the leaf index is the loop variable, so nothing is decoded per
// candidate, the multi-word peel happens once per several thousand of them, and the dictionary
// read is uniform across the block again.
__global__ __launch_bounds__(256)
void MitmBackward(const Word* __restrict__ dict, PosDict pd,
                  uint32_t splitAt, uint32_t depth,
                  const uint64_t* __restrict__ targets, uint32_t targetCount,
                  uint64_t begin, uint64_t end,
                  const uint64_t* __restrict__ keys, const uint32_t* __restrict__ vals,
                  uint32_t tableMask,
                  uint64_t* __restrict__ hitFwd, uint64_t* __restrict__ hitBwd,
                  uint32_t* __restrict__ hitCount, uint32_t hitMax)
{
    for (uint64_t stem = begin + blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
         stem < end;
         stem += (uint64_t)gridDim.x * blockDim.x) {

        uint32_t widx[MAX_DEPTH];
        uint64_t x = stem;
        for (uint32_t p = splitAt + 1; p < depth; p++) {
            uint32_t n = pd.count[p];
            widx[p] = (uint32_t)(x % n);
            x /= n;
        }
        uint32_t top = (uint32_t)(x % 16);
        uint32_t t = (uint32_t)(x / 16);
        if (t >= targetCount) continue;

        uint64_t h = targets[t] | ((uint64_t)top << 60);

        // Peel the last word, then a separator and a word for each one further back, then the
        // separator that sits between the leaf and the rest.
        bool any = false;
        for (uint32_t p = depth; p-- > splitAt + 1; ) {
            if (any) h = FnvUnstep(h, '_');
            any = true;
            Word w = dict[pd.base[p] + widx[p]];
            h = UnhashWord(h, w, (w.b.w >> 24) & 0xFF);
        }
        if (any) h = FnvUnstep(h, '_');

        // Four leaves at a time, and their four table reads issued before any is examined.
        //
        // This loop is not limited by arithmetic - it runs at under two percent of the card's
        // integer throughput - but by the latency of one dependent random read into an 800 MB
        // table. Nothing else can start while a thread waits on it. Computing four hashes and
        // then loading four slots puts four reads in flight per thread instead of one, which is
        // the only lever left once the work itself cannot be made smaller.
        const uint32_t leafBase = pd.base[splitAt];
        const uint32_t leaves = pd.count[splitAt];
        for (uint32_t j = 0; j < leaves; j += 4) {
            uint64_t s[4];
            uint32_t slot[4];
            uint64_t k[4];
            uint32_t n = min(4u, leaves - j);

            for (uint32_t q = 0; q < n; q++) {
                Word w = dict[leafBase + j + q];
                s[q] = UnhashWord(h, w, (w.b.w >> 24) & 0xFF);
                slot[q] = (uint32_t)(s[q] ^ (s[q] >> 32)) & tableMask;
            }
            for (uint32_t q = 0; q < n; q++) k[q] = keys[slot[q]];

            for (uint32_t q = 0; q < n; q++) {
                uint64_t key = k[q];
                uint32_t sl = slot[q];
                while (key != TABLE_EMPTY) {          // collisions fall into this slow path only
                    if (key == s[q]) {
                        uint32_t o = atomicAdd(hitCount, 1u);
                        if (o < hitMax) { hitFwd[o] = vals[sl]; hitBwd[o] = stem * leaves + j + q; }
                        break;
                    }
                    sl = (sl + 1) & tableMask;
                    key = keys[sl];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Host

struct Options {
    std::string targets, dict, prefix, prefixFile, prefixTable, out;
    int depthMin = 1, depthMax = 3;
    int mitm = 0;                 // --mitm K: meet in the middle, forward over the first K words
    int bitmapBits = 17;          // 2^17 bits = 16 KB, which still lets six blocks sit on an SM
    // Stems per launch. Sized for a headless Linux box, where a kernel may run as long as it
    // likes: fewer, larger launches means less launch overhead and fewer host synchronisations.
    // Lower it to about 1<<20 on Windows, or anywhere the GPU also drives a display - there a
    // kernel that outruns the driver's timeout is killed, along with the process, after two
    // seconds by default.
    long long chunk = 1LL << 23;
};

static std::vector<uint64_t> ReadTargets(const std::string& path)
{
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(1); }

    std::vector<uint64_t> out;
    std::string line;
    while (std::getline(in, line)) {
        // Accept both the bare hex and the `hash_<hex>` spelling acts uses, so the same target
        // file feeds either tool.
        size_t at = line.rfind('_');
        std::string hex = (at == std::string::npos) ? line : line.substr(at + 1);
        while (!hex.empty() && std::isspace((unsigned char)hex.back())) hex.pop_back();
        if (hex.empty()) continue;

        char* end = nullptr;
        uint64_t v = std::strtoull(hex.c_str(), &end, 16);
        if (end && *end == 0) out.push_back(v & MASK60);
    }

    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

static std::vector<std::string> ReadDict(const std::string& path, size_t& dropped)
{
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(1); }

    std::vector<std::string> out;
    std::string line;
    dropped = 0;
    while (std::getline(in, line)) {
        while (!line.empty() && std::isspace((unsigned char)line.back())) line.pop_back();
        if (line.empty()) continue;
        // A word past the stride cannot be a single aligned load. Dropping it is reported rather
        // than silent, because a missing word is a name the search can never reach.
        if (line.size() > WORD_MAX) { dropped++; continue; }
        for (auto& c : line) c = (char)std::tolower((unsigned char)c);
        out.push_back(line);
    }
    return out;
}

// Mirrors the kernel's mixed-radix decoding exactly. Any disagreement between the two shows up as
// a name that does not hash back to its target, which main() prints as rejected.
static std::string Rebuild(uint64_t index, const std::vector<std::vector<std::string>>& lists,
                           const std::string& prefix, uint32_t stemDepth)
{
    uint32_t leaves = (uint32_t)lists[stemDepth].size();
    uint32_t leaf = (uint32_t)(index % leaves);
    uint64_t stem = index / leaves;

    std::string out = prefix;
    for (uint32_t k = 0; k < stemDepth; k++) {
        uint32_t n = (uint32_t)lists[k].size();
        out += lists[k][stem % n];
        out += '_';
        stem /= n;
    }
    out += lists[stemDepth][leaf];
    return out;
}

// Rebuilds a name from the pair the meet-in-the-middle returns. Mirrors both kernels' index
// arithmetic; a disagreement shows up as a name that does not hash to a target, which main()
// prints as rejected rather than reporting.
static std::string RebuildMitm(uint64_t fwd, uint64_t bwd,
                               const std::vector<std::vector<std::string>>& lists,
                               const std::string& prefix, uint32_t splitAt, uint32_t depth)
{
    std::string out = prefix;
    uint64_t x = fwd;
    for (uint32_t k = 0; k < splitAt; k++) {
        uint32_t n = (uint32_t)lists[k].size();
        out += lists[k][x % n];
        out += '_';
        x /= n;
    }

    // The backward index is the stem times the leaf count plus the leaf, and the stem itself is
    // mixed radix over the positions after the split - the same order the kernel builds it in.
    uint32_t leaves = (uint32_t)lists[splitAt].size();
    uint64_t leaf = bwd % leaves;
    uint64_t stem = bwd / leaves;

    uint32_t widx[MAX_DEPTH]{};
    for (uint32_t p = splitAt + 1; p < depth; p++) {
        uint32_t n = (uint32_t)lists[p].size();
        widx[p] = (uint32_t)(stem % n);
        stem /= n;
    }

    out += lists[splitAt][leaf];
    for (uint32_t p = splitAt + 1; p < depth; p++) {
        out += '_';
        out += lists[p][widx[p]];
    }
    return out;
}

static std::vector<std::string> SplitPaths(const std::string& s)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string one;
    while (std::getline(ss, one, ',')) if (!one.empty()) out.push_back(one);
    return out;
}

int main(int argc, char** argv)
{
    Options o;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        bool next = i + 1 < argc;
        if      (a == "--targets" && next) o.targets = argv[++i];
        else if (a == "--dict" && next)    o.dict = argv[++i];
        else if (a == "--prefix" && next)  o.prefix = argv[++i];
        else if (a == "--prefix-file" && next) o.prefixFile = argv[++i];
        else if (a == "--prefix-table" && next) o.prefixTable = argv[++i];
        else if (a == "--out" && next)     o.out = argv[++i];
        else if (a == "--depth" && next)   o.depthMax = std::atoi(argv[++i]);
        else if (a == "--depth-min" && next) o.depthMin = std::atoi(argv[++i]);
        else if (a == "--mitm" && next)    o.mitm = std::atoi(argv[++i]);
        else if (a == "--bitmap-bits" && next) o.bitmapBits = std::atoi(argv[++i]);
        else if (a == "--chunk" && next)   o.chunk = std::atoll(argv[++i]);
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (o.targets.empty() || o.dict.empty()) {
        std::fprintf(stderr,
            "usage: codehash --targets <file> --dict <file> [--prefix S | --prefix-file F]\n"
            "                [--depth N] [--depth-min N] [--bitmap-bits N] [--chunk N] [--out F]\n");
        return 2;
    }

    std::vector<uint64_t> targets = ReadTargets(o.targets);

    // --dict takes one file per position, comma separated, in order. A single file is used for
    // every position, which is the flat search; several files is the positional one. If fewer
    // files than the depth are given the last repeats, so `pos0,pos1,rest` is a valid shorthand.
    size_t dropped = 0;
    std::vector<std::vector<std::string>> lists;
    for (const std::string& p : SplitPaths(o.dict)) lists.push_back(ReadDict(p, dropped));
    if (targets.empty() || lists.empty() || lists[0].empty()) {
        std::fprintf(stderr, "empty targets or dictionary\n"); return 2;
    }
    while ((int)lists.size() < o.depthMax) lists.push_back(lists.back());

    // A run is split by prefix because every level not written into the prefix has to come out of
    // the dictionary, and each one multiplies the space by the vocabulary size. Fixing the map in
    // the prefix - `ui_icon_inventory_zm_red_` rather than `ui_icon_inventory_` - spares two
    // levels, `zm` and `red`, which is a factor of twenty-five million against a cost of eight
    // runs. Those eight then belong in ONE process: a CUDA context costs about two hundred
    // milliseconds to build, which is a quarter of the wall time of a fast pass.
    std::vector<std::string> prefixes;
    if (!o.prefixFile.empty()) {
        std::ifstream in(o.prefixFile);
        if (!in) { std::fprintf(stderr, "cannot read %s\n", o.prefixFile.c_str()); return 1; }
        std::string line;
        while (std::getline(in, line)) {
            // Trailing carriage returns are the failure this program is most likely to meet: a
            // prefix list written on Windows hashes "..._\r", matches nothing, and says nothing.
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
            if (!line.empty()) prefixes.push_back(line);
        }
    }
    if (!o.prefix.empty() || prefixes.empty()) prefixes.push_back(o.prefix);

    std::printf("targets %zu, %zu position(s):", targets.size(), lists.size());
    for (size_t i = 0; i < lists.size() && i < MAX_DEPTH; i++) std::printf(" %zu", lists[i].size());
    if (dropped) std::printf("  (%zu dropped, longer than %d characters)", dropped, WORD_MAX);
    std::printf("\n%zu prefix(es)\n", prefixes.size());

    // Sizing the bitmap is a trade against occupancy, not just against false positives: shared
    // memory per SM is 100 KB, so 16 KB leaves room for six blocks and full occupancy, while
    // 64 KB would leave room for one. A bitmap false positive costs a binary search, nothing more.
    uint32_t bitmapBits = 1u << o.bitmapBits;
    uint32_t bitmapWords = bitmapBits >> 5;
    size_t   smemBytes = (size_t)bitmapWords * sizeof(uint32_t);
    std::vector<uint32_t> bitmap(bitmapWords, 0u);
    for (uint64_t h : targets) {
        uint32_t bit = (uint32_t)(h & (bitmapBits - 1));
        bitmap[bit >> 5] |= 1u << (bit & 31);
    }
    double fp = (double)targets.size() / (double)bitmapBits;
    std::printf("bitmap %u bits (%.0f KB shared), false positive rate %.3f%%\n",
                bitmapBits, smemBytes / 1024.0, fp * 100.0);

    // Every position's list packed end to end into one array, one 16-byte aligned entry per word
    // with the length in the last byte. PosDict then just holds where each position starts.
    PosDict pd{};
    std::vector<Word> packed;
    for (size_t p = 0; p < lists.size() && p < MAX_DEPTH; p++) {
        pd.base[p] = (uint32_t)packed.size();
        pd.count[p] = (uint32_t)lists[p].size();
        for (const std::string& w : lists[p]) {
            Word e{};
            std::memcpy(&e, w.data(), w.size());
            ((uint8_t*)&e)[WORD_STRIDE - 1] = (uint8_t)w.size();
            packed.push_back(e);
        }
    }

    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("device %s, %d SMs, L2 %d MB, persisting L2 max %zu KB\n",
                prop.name, prop.multiProcessorCount, prop.l2CacheSize >> 20,
                (size_t)prop.persistingL2CacheMaxSize >> 10);

    Word*     dDict = nullptr;
    uint32_t* dBitmap = nullptr;
    uint64_t* dTable = nullptr;
    uint64_t* dHits = nullptr;
    uint32_t* dHitCount = nullptr;
    // Sixty-five thousand was enough while a run returned dozens. Recombining whole prefix and
    // suffix tables returns tens of thousands, and a full buffer silently drops the overflow -
    // one run reported 80815 matches and kept 65536. Sixteen megabytes of slack costs nothing
    // next to losing a quarter of the answers.
    const uint32_t hitMax = 1u << 21;

    CUDA_CHECK(cudaMalloc(&dDict, packed.size() * sizeof(Word)));
    CUDA_CHECK(cudaMalloc(&dBitmap, bitmap.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dTable, targets.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&dHits, (size_t)hitMax * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&dHitCount, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(dDict, packed.data(), packed.size() * sizeof(Word), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBitmap, bitmap.data(), bitmap.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dTable, targets.data(), targets.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Pin the sorted table in L2. Only the candidates that clear the bitmap read it, but when they
    // do it should not be a trip to DRAM - and the table is small enough that reserving room for
    // it costs the rest of the kernel nothing.
    size_t tableBytes = targets.size() * sizeof(uint64_t);
    size_t persistBytes = std::min<size_t>(tableBytes, (size_t)prop.persistingL2CacheMaxSize);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    if (persistBytes > 0) {
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persistBytes));
        cudaStreamAttrValue attr{};
        attr.accessPolicyWindow.base_ptr  = dTable;
        attr.accessPolicyWindow.num_bytes = persistBytes;
        attr.accessPolicyWindow.hitRatio  = 1.0f;
        attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
        CUDA_CHECK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
        std::printf("L2 persistence on %zu KB of hash table\n", persistBytes >> 10);
    }

    // Above 48 KB a kernel has to ask for its shared memory explicitly.
    if (smemBytes > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(Search, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemBytes));
    }

    const int block = 256;
    int blocksPerSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, Search, block, smemBytes));
    if (blocksPerSM < 1) blocksPerSM = 1;
    int grid = prop.multiProcessorCount * blocksPerSM;
    std::printf("grid %d blocks of %d, %d blocks per SM (%d%% occupancy)\n\n",
                grid, block, blocksPerSM,
                (int)(100.0 * blocksPerSM * block / prop.maxThreadsPerMultiProcessor));

    std::vector<std::string> found;

    // Prefix discovery: instead of being told which prefix to try, walk backwards from every
    // target through the tail words and see which KNOWN prefix the walk lands on.
    //
    // This exists because ranking prefixes was the weakest part of the search. A prefix was scored
    // by how many named assets sit under it, which is exactly backwards - the names still missing
    // are the ones whose family nobody has documented, so the ranking pushed them down. Here the
    // whole corpus of prefixes goes into the table at once, half a million of them, and the search
    // reports which one each target belongs to. No ranking, no blind spot.
    //
    // The table stays around six megabytes, so it sits in L2 and the backward pass runs at full
    // speed - the same lesson as before, arrived at from the other side.
    if (!o.prefixTable.empty()) {
        std::ifstream in(o.prefixTable);
        if (!in) { std::fprintf(stderr, "cannot read %s\n", o.prefixTable.c_str()); return 1; }

        std::vector<std::string> table;
        std::string line;
        while (std::getline(in, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
            if (!line.empty()) table.push_back(line);
        }
        if (table.empty()) { std::fprintf(stderr, "empty prefix table\n"); return 2; }

        const uint32_t depth = (uint32_t)o.depthMax;
        if (depth < 1 || depth > (uint32_t)lists.size()) {
            std::fprintf(stderr, "--depth must be between 1 and the number of position lists\n");
            return 2;
        }

        uint32_t tableSize = 1u;
        while ((uint64_t)tableSize < table.size() * 2ull) tableSize <<= 1;
        const size_t tableBytes = (size_t)tableSize * (sizeof(uint64_t) + sizeof(uint32_t));

        uint64_t bwdStemTuples = 1;
        for (uint32_t p = 1; p < depth; p++) bwdStemTuples *= pd.count[p];
        const uint64_t bwdStems = bwdStemTuples * 16ull * targets.size();

        std::printf("prefix table: %zu (%.1f MB%s), depth %u, %llu backward stems\n",
                    table.size(), tableBytes / 1e6,
                    tableBytes <= (size_t)prop.l2CacheSize ? ", fits in L2" : ", EXCEEDS L2",
                    depth, (unsigned long long)bwdStems);

        std::vector<uint64_t> hostKeys(tableSize, TABLE_EMPTY);
        std::vector<uint32_t> hostVals(tableSize, 0u);
        for (uint32_t i = 0; i < (uint32_t)table.size(); i++) {
            const uint64_t h = FnvString(FNV_OFFSET, table[i].data(), table[i].size());
            uint32_t slot = (uint32_t)(h ^ (h >> 32)) & (tableSize - 1);
            while (hostKeys[slot] != TABLE_EMPTY && hostKeys[slot] != h)
                slot = (slot + 1) & (tableSize - 1);
            if (hostKeys[slot] == TABLE_EMPTY) { hostKeys[slot] = h; hostVals[slot] = i; }
        }

        uint64_t *dKeys = nullptr, *dHitF = nullptr, *dHitB = nullptr;
        uint32_t* dVals = nullptr;
        CUDA_CHECK(cudaMalloc(&dKeys, (size_t)tableSize * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&dVals, (size_t)tableSize * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dHitF, (size_t)hitMax * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&dHitB, (size_t)hitMax * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(dKeys, hostKeys.data(), (size_t)tableSize * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dVals, hostVals.data(), (size_t)tableSize * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dHitCount, 0, sizeof(uint32_t)));

        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0));
        CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0, stream));

        for (uint64_t begin = 0; begin < bwdStems; begin += (uint64_t)o.chunk) {
            uint64_t end = std::min(bwdStems, begin + (uint64_t)o.chunk);
            MitmBackward<<<grid, block, 0, stream>>>(dDict, pd, 0, depth,
                                                     dTable, (uint32_t)targets.size(),
                                                     begin, end, dKeys, dVals, tableSize - 1,
                                                     dHitF, dHitB, dHitCount, hitMax);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::printf("\r  %5.1f%%", 100.0 * (double)end / (double)bwdStems);
            std::fflush(stdout);
        }
        CUDA_CHECK(cudaEventRecord(t1, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

        uint32_t count = 0;
        CUDA_CHECK(cudaMemcpy(&count, dHitCount, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        uint32_t take = std::min(count, hitMax);
        std::vector<uint64_t> hf(take), hb(take);
        if (take) {
            CUDA_CHECK(cudaMemcpy(hf.data(), dHitF, take * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hb.data(), dHitB, take * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        }
        std::printf("\r  %.1f s, %u hit(s)%s\n", ms / 1000.0, count,
                    count > hitMax ? " (buffer plein)" : "");

        std::ofstream out;
        if (!o.out.empty()) out.open(o.out, std::ios::app);
        size_t kept = 0;
        for (uint32_t i = 0; i < take; i++) {
            if (hf[i] >= table.size()) continue;
            std::string name = RebuildMitm(0, hb[i], lists, table[(size_t)hf[i]], 0, depth);
            uint64_t h = FnvString(FNV_OFFSET, name.data(), name.size()) & MASK60;
            if (!std::binary_search(targets.begin(), targets.end(), h)) continue;
            std::printf("    %s\n", name.c_str());
            if (out) out << name << "\n";
            kept++;
        }
        std::printf("%zu name(s) kept\n", kept);

        cudaFree(dKeys); cudaFree(dVals); cudaFree(dHitF); cudaFree(dHitB);
        cudaFree(dDict); cudaFree(dBitmap); cudaFree(dTable); cudaFree(dHits); cudaFree(dHitCount);
        cudaStreamDestroy(stream);
        return 0;
    }

    if (o.mitm > 0) {
        const uint32_t depth = (uint32_t)o.depthMax;
        const uint32_t splitAt = (uint32_t)o.mitm;
        if (splitAt >= depth || depth > (uint32_t)lists.size()) {
            std::fprintf(stderr, "--mitm must be below --depth, and enough position lists given\n");
            return 2;
        }

        uint64_t fwdCount = 1, bwdTuples = 1, bwdStemTuples = 1;
        for (uint32_t k = 0; k < splitAt; k++) fwdCount *= pd.count[k];
        for (uint32_t p = splitAt; p < depth; p++) bwdTuples *= pd.count[p];
        for (uint32_t p = splitAt + 1; p < depth; p++) bwdStemTuples *= pd.count[p];
        const uint64_t bwdCount = bwdTuples * 16ull * targets.size();
        const uint64_t bwdStems = bwdStemTuples * 16ull * targets.size();

        // The forward half is what has to be held, so it is the half that must be small: sizing
        // the split is a memory decision before it is a speed one.
        uint32_t tableSize = 1u;
        while ((uint64_t)tableSize < fwdCount * 2ull) tableSize <<= 1;
        const size_t tableBytes = (size_t)tableSize * (sizeof(uint64_t) + sizeof(uint32_t));

        std::printf("MITM: %llu forward states (table %u slots, %.0f MB), %llu backward candidates\n",
                    (unsigned long long)fwdCount, tableSize, tableBytes / 1e6,
                    (unsigned long long)bwdCount);

        // Choose the split so this table fits in L2, not merely in VRAM. Every backward candidate
        // probes it once at a random address, so the whole pass runs at the speed of that lookup.
        // Measured on a 3090 at depth four: an 805 MB table gives 5.2e9 candidates a second, 25%
        // of DRAM bandwidth and an 8% L2 hit rate; the same work against a 2 MB table gives
        // 3.4e10 - six and a half times faster for an identical candidate count. Positional
        // vocabularies are what make the forward half small enough to get there.
        if (tableBytes > (size_t)prop.l2CacheSize) {
            std::printf("      WARNING: the table exceeds L2 (%d MB). Every probe reaches DRAM and the\n"
                        "      backward pass runs about 6x slower. Move the split, or give shorter\n"
                        "      positional lists for the leading positions.\n",
                        prop.l2CacheSize >> 20);
        }
        std::printf("      contre %.3g en recherche directe  (/%.0f)\n",
                    (double)fwdCount * (double)bwdTuples * 1.0,
                    ((double)fwdCount * (double)bwdTuples) / (double)(fwdCount + bwdCount));

        uint64_t* dKeys = nullptr;
        uint32_t* dVals = nullptr;
        uint64_t* dHitF = nullptr;
        uint64_t* dHitB = nullptr;
        CUDA_CHECK(cudaMalloc(&dKeys, (size_t)tableSize * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&dVals, (size_t)tableSize * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dHitF, (size_t)hitMax * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&dHitB, (size_t)hitMax * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(dKeys, 0xFF, (size_t)tableSize * sizeof(uint64_t)));

        for (const std::string& prefix : prefixes) {
            uint64_t prefixState = FnvString(FNV_OFFSET, prefix.data(), prefix.size());
            std::printf("prefix \"%s\"\n", prefix.c_str());

            CUDA_CHECK(cudaMemset(dKeys, 0xFF, (size_t)tableSize * sizeof(uint64_t)));
            CUDA_CHECK(cudaMemset(dHitCount, 0, sizeof(uint32_t)));

            cudaEvent_t t0, t1;
            CUDA_CHECK(cudaEventCreate(&t0));
            CUDA_CHECK(cudaEventCreate(&t1));
            CUDA_CHECK(cudaEventRecord(t0, stream));

            MitmForward<<<grid, block, 0, stream>>>(dDict, pd, prefixState, splitAt, fwdCount,
                                                    dKeys, dVals, tableSize - 1);
            CUDA_CHECK(cudaGetLastError());

            for (uint64_t begin = 0; begin < bwdStems; begin += (uint64_t)o.chunk) {
                uint64_t end = std::min(bwdStems, begin + (uint64_t)o.chunk);
                MitmBackward<<<grid, block, 0, stream>>>(dDict, pd, splitAt, depth,
                                                         dTable, (uint32_t)targets.size(),
                                                         begin, end, dKeys, dVals, tableSize - 1,
                                                         dHitF, dHitB, dHitCount, hitMax);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaStreamSynchronize(stream));
                std::printf("\r  %5.1f%%", 100.0 * (double)end / (double)bwdStems);
                std::fflush(stdout);
            }

            CUDA_CHECK(cudaEventRecord(t1, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

            uint32_t count = 0;
            CUDA_CHECK(cudaMemcpy(&count, dHitCount, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            uint32_t take = std::min(count, hitMax);
            std::vector<uint64_t> hf(take), hb(take);
            if (take) {
                CUDA_CHECK(cudaMemcpy(hf.data(), dHitF, take * sizeof(uint64_t), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hb.data(), dHitB, take * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            }
            std::printf("\r  %.1f s, %.3g operations/s, %u hit(s)\n",
                        ms / 1000.0, (double)(fwdCount + bwdCount) / (ms / 1000.0), count);

            for (uint32_t i = 0; i < take; i++) {
                std::string name = RebuildMitm(hf[i], hb[i], lists, prefix, splitAt, depth);
                uint64_t h = FnvString(FNV_OFFSET, name.data(), name.size()) & MASK60;
                if (!std::binary_search(targets.begin(), targets.end(), h)) {
                    std::printf("    [rejete] %s\n", name.c_str());
                    continue;
                }
                std::printf("    %s\n", name.c_str());
                found.push_back(name);
            }

            CUDA_CHECK(cudaEventDestroy(t0));
            CUDA_CHECK(cudaEventDestroy(t1));
        }

        cudaFree(dKeys); cudaFree(dVals); cudaFree(dHitF); cudaFree(dHitB);

        if (!o.out.empty() && !found.empty()) {
            std::ofstream out(o.out, std::ios::app);
            for (const auto& n : found) out << n << "\n";
            std::printf("\n%zu name(s) written to %s\n", found.size(), o.out.c_str());
        }
        cudaFree(dDict); cudaFree(dBitmap); cudaFree(dTable); cudaFree(dHits); cudaFree(dHitCount);
        cudaStreamDestroy(stream);
        return 0;
    }

    for (const std::string& prefix : prefixes) {
    uint64_t prefixState = FnvString(FNV_OFFSET, prefix.data(), prefix.size());
    std::printf("prefix \"%s\"\n", prefix.c_str());

    for (int depth = std::max(1, o.depthMin); depth <= o.depthMax; depth++) {
        uint32_t stemDepth = (uint32_t)(depth - 1);

        if (depth > (int)lists.size() || depth > MAX_DEPTH) {
            std::printf("depth %d: only %zu position list(s) given, stopping\n", depth, lists.size());
            break;
        }

        // Stems are the product of the sizes of every position before the leaf - not a power,
        // since each position has its own vocabulary.
        uint64_t stems = 1;
        bool overflow = false;
        for (uint32_t k = 0; k < stemDepth; k++) {
            uint32_t n = pd.count[k];
            if (n == 0 || stems > (UINT64_MAX / n)) { overflow = true; break; }
            stems *= n;
        }
        if (overflow) { std::printf("depth %d: search space overflows 64 bits, stopping\n", depth); break; }

        double candidates = (double)stems * (double)pd.count[stemDepth];
        std::printf("depth %d: %.3g candidates, ~%.1f expected false hits\n",
                    depth, candidates, candidates * (double)targets.size() / 1.152921504606847e18);

        CUDA_CHECK(cudaMemset(dHitCount, 0, sizeof(uint32_t)));

        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0));
        CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0, stream));

        for (uint64_t begin = 0; begin < stems; begin += (uint64_t)o.chunk) {
            uint64_t end = std::min(stems, begin + (uint64_t)o.chunk);
            Search<<<grid, block, smemBytes, stream>>>(
                dDict, pd, prefixState, stemDepth, begin, end,
                dBitmap, bitmapWords, bitmapBits - 1,
                dTable, (uint32_t)targets.size(),
                dHits, dHitCount, hitMax);
            CUDA_CHECK(cudaGetLastError());

            if (stems > (uint64_t)o.chunk) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                std::printf("\r  %5.1f%%", 100.0 * (double)end / (double)stems);
                std::fflush(stdout);
            }
        }
        CUDA_CHECK(cudaEventRecord(t1, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

        uint32_t count = 0;
        CUDA_CHECK(cudaMemcpy(&count, dHitCount, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        uint32_t take = std::min(count, hitMax);
        std::vector<uint64_t> hits(take);
        if (take) CUDA_CHECK(cudaMemcpy(hits.data(), dHits, take * sizeof(uint64_t), cudaMemcpyDeviceToHost));

        std::printf("\r  %.1f s, %.3g candidats/s, %u hit(s)%s\n",
                    ms / 1000.0, candidates / (ms / 1000.0), count,
                    count > hitMax ? " (buffer plein)" : "");

        for (uint64_t idx : hits) {
            std::string name = Rebuild(idx, lists, prefix, stemDepth);
            // Re-derive on the host rather than trust the device: a name that does not hash back
            // to a target is a bug in this program, not a discovery, and it should be visible.
            uint64_t h = FnvString(FNV_OFFSET, name.data(), name.size()) & MASK60;
            if (!std::binary_search(targets.begin(), targets.end(), h)) {
                std::printf("    [rejete] %s\n", name.c_str());
                continue;
            }
            std::printf("    %s\n", name.c_str());
            found.push_back(name);

            // Appended and flushed as it is found, not held to the end. A sweep over thousands of
            // prefixes runs for hours, and everything it has proved should survive being stopped.
            if (!o.out.empty()) {
                std::ofstream out(o.out, std::ios::app);
                out << name << "\n";
            }
        }

        CUDA_CHECK(cudaEventDestroy(t0));
        CUDA_CHECK(cudaEventDestroy(t1));
    }
    }

    if (!o.out.empty() && !found.empty())
        std::printf("\n%zu name(s) written to %s\n", found.size(), o.out.c_str());

    cudaFree(dDict); cudaFree(dBitmap); cudaFree(dTable); cudaFree(dHits); cudaFree(dHitCount);
    cudaStreamDestroy(stream);
    return 0;
}
