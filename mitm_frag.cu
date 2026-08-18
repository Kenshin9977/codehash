// mitm_frag - meet in the middle over observed name fragments.
//
// A different search from codehash.cu, and worth stating why rather than folding it in.
//
// codehash enumerates names as combinations of dictionary WORDS, which is the right shape while a
// name is short. It stops being the right shape quickly: names in this corpus have a median of
// nine words, and nine words drawn from a fifty thousand word vocabulary is 2^140 combinations
// against 2^60 hashes. Past about four words there are more candidate names than there are hashes,
// so arithmetic cannot pick the true one out - around 2^80 word sequences hash to any given
// target, and every one of them is an equally valid preimage. A faster search does not help. Only
// a better prior does.
//
// So this searches fragments instead of words. Every prefix and every suffix cut at a separator
// boundary out of the names already known becomes a candidate half. A missing name that is a
// recombination - a known prefix meeting a known suffix - is then found by a meet in the middle,
// and the space searched is the product of two observed sets rather than a vocabulary raised to a
// power.
//
// Measured on the first run: 883 488 prefixes against 3 107 168 suffixes over 50 635 targets is
// 2.7e12 pairs, and it returned 641 distinct names in 19 seconds, every one of which verified.
// The expected number of false matches for that whole run was 0.12, which is the useful property
// of a search this shape - it is precise by construction, and what it needs is reach.
//
//   nvcc -O3 -arch=sm_86 -o mitm_frag mitm_frag.cu
//   mitm_frag --prefixes p.txt --suffixes s.txt --targets t.txt [--extend w.txt] --out found.txt

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// Everything runs modulo 2^60, and that is the point rather than a detail.
//
// The stored hash is FNV-1a truncated to 60 bits, so the top four bits of the state are gone. The
// obvious reading is that a backward pass has to try all sixteen of them, and the sibling program
// does exactly that: it walks 16 * targets * stems. It does not have to. Multiplication modulo
// 2^64 truncated to 60 bits equals multiplication modulo 2^60 of the truncated operands, and XOR
// with a byte cannot reach bit 60, so the whole chain - forward and backward - is closed under
// truncation. Checked against the full 64-bit computation over three thousand random splits.
//
// That is a free factor of sixteen on the backward pass.
#define MASK60   0x0FFFFFFFFFFFFFFFULL
#define FNV_OFF  0xCBF29CE484222325ULL
#define FNV_P    0x100000001B3ULL
#define FNV_PINV 0x0E965057AFF6957BULL   // P^-1 mod 2^60
#define EMPTY    0xFFFFFFFFFFFFFFFFULL

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "cuda: %s at line %d\n", cudaGetErrorString(e_), __LINE__); \
    std::exit(1); } } while (0)

__device__ __forceinline__ uint64_t StepBack(uint64_t h, uint32_t b)
{
    return ((h * FNV_PINV) & MASK60) ^ (uint64_t)b;
}

__device__ __forceinline__ uint32_t Slot(uint64_t h, uint32_t mask)
{
    return ((uint32_t)(h ^ (h >> 29)) * 2654435761u) & mask;
}

// Open addressing, linear probing, keyed on the 60-bit forward state. Empty is all ones, which a
// real state matches only by having all sixty bits set - one chance in 2^60, and the cost of
// losing that bet is one missed name out of hundreds of thousands.
__device__ __forceinline__ int64_t Lookup(const uint64_t* __restrict__ keys,
                                          const uint64_t* __restrict__ vals,
                                          uint32_t mask, uint64_t h)
{
    uint32_t slot = Slot(h, mask);
    for (;;) {
        const uint64_t k = keys[slot];
        if (k == h)     return (int64_t)vals[slot];
        if (k == EMPTY) return -1;
        slot = (slot + 1) & mask;
    }
}

// Build the forward table on the device.
//
// The two halves are not symmetric and it pays to know which. A prefix costs sixteen bytes of
// table and nothing else; a suffix costs a full backward walk against every target, so it costs
// time. Growing the prefix side is therefore the cheap direction, and that is where extra reach is
// bought: every observed prefix, and every observed prefix with one more word glued on. That
// reaches names shaped (known prefix)(one new word)(known suffix), which a pure recombination
// search cannot.
//
// Which words may follow is a per-prefix question, not a global one.
//
// Extending every prefix by the same commonest words spends the budget on things nobody would
// write - `i_c_t8_mp_spe_` followed by `wav`, say. The budget matters because the binding limit
// here is not memory or time but false matches: pairs * targets / 2^60 grows with the search and
// a wrong name that lands on a target hash is indistinguishable from a right one. So the words
// offered to each prefix are the ones that have actually followed its last token somewhere in the
// corpus, capped and ordered by frequency. Same pair count, far better spent: 109 continuations
// per prefix chosen this way against 256 generic ones costs 13 expected false matches instead of
// 31, and reaches names the generic list never proposes.
//
// One thread per prefix, so the prefix is hashed once and every continuation extends that state
// rather than restarting from the offset basis.
__global__ void Build(const char* __restrict__ pText, const uint32_t* __restrict__ pOff,
                      uint32_t pCount,
                      const char* __restrict__ wText, const uint32_t* __restrict__ wOff,
                      const uint32_t* __restrict__ contOff, const uint32_t* __restrict__ contIdx,
                      uint64_t* __restrict__ keys, uint64_t* __restrict__ vals, uint32_t mask)
{
    for (uint32_t p = blockIdx.x * blockDim.x + threadIdx.x; p < pCount;
         p += gridDim.x * blockDim.x) {

        uint64_t base = FNV_OFF & MASK60;
        for (uint32_t i = pOff[p]; i < pOff[p + 1]; i++)
            base = ((base ^ (uint64_t)(unsigned char)pText[i]) * FNV_P) & MASK60;

        const uint32_t first = contOff ? contOff[p] : 0u;
        const uint32_t last  = contOff ? contOff[p + 1] : 0u;

        for (uint32_t k = 0; k <= last - first; k++) {
            uint64_t h = base;
            if (k) {
                const uint32_t w = contIdx[first + k - 1];
                for (uint32_t i = wOff[w]; i < wOff[w + 1]; i++)
                    h = ((h ^ (uint64_t)(unsigned char)wText[i]) * FNV_P) & MASK60;
                h = ((h ^ (uint64_t)'_') * FNV_P) & MASK60;
            }

            // Twenty-four bits for the continuation slot. It has overflowed twice: eight bits
            // lost 355 names of 8125 at a cap of 512, and sixteen lost 49 of 15 562 once the
            // commonest prefixes were handed the whole 167 631 word vocabulary. Both times the
            // wrong string still matched a target state, so nothing but recomputing the hash
            // could tell. Twenty-four bits leaves room for a vocabulary sixty times larger, and
            // a prefix index of twenty bits alongside it still fits a uint64 twice over.
            const uint64_t id = ((uint64_t)p << 24) | (uint64_t)k;
            uint32_t slot = Slot(h, mask);
            for (;;) {
                const uint64_t old = atomicCAS((unsigned long long*)&keys[slot],
                                               (unsigned long long)EMPTY, (unsigned long long)h);
                if (old == EMPTY) { vals[slot] = id; break; }
                if (old == h)     break;      // same state twice; the first writer keeps the slot
                slot = (slot + 1) & mask;
            }
        }
    }
}

// One block per suffix, threads spread across the targets.
//
// The suffix characters are then read once and shared by the whole block, and every thread in a
// warp walks the same number of steps, because the inversion length is a property of the suffix
// and not of the target. Nothing diverges.
__global__ __launch_bounds__(256)
void Search(const char* __restrict__ text, const uint32_t* __restrict__ offset,
            uint32_t suffixCount,
            const uint64_t* __restrict__ targets, uint32_t targetCount,
            const uint64_t* __restrict__ keys, const uint64_t* __restrict__ vals, uint32_t mask,
            uint64_t* __restrict__ hits, uint32_t* __restrict__ hitCount, uint32_t hitMax)
{
    for (uint32_t s = blockIdx.x; s < suffixCount; s += gridDim.x) {
        const uint32_t begin = offset[s];
        const uint32_t len   = offset[s + 1] - begin;

        for (uint32_t t = threadIdx.x; t < targetCount; t += blockDim.x) {
            uint64_t h = targets[t];
            for (int32_t i = (int32_t)len - 1; i >= 0; i--)
                h = StepBack(h, (uint32_t)(unsigned char)text[begin + i]);

            const int64_t id = Lookup(keys, vals, mask, h);
            if (id >= 0) {
                const uint32_t slot = atomicAdd(hitCount, 1u);
                if (slot < hitMax) {
                    hits[slot * 2 + 0] = (uint64_t)id;
                    hits[slot * 2 + 1] = s;
                }
            }
        }
    }
}

static std::vector<std::string> ReadLines(const std::string& path)
{
    std::vector<std::string> out;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        // Trailing carriage returns are the failure this kind of program meets most often: a list
        // written on Windows hashes one extra byte, matches nothing, and says nothing about why.
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

static std::vector<uint32_t> ReadU32(const std::string& path)
{
    std::vector<uint32_t> out;
    if (path.empty()) return out;

    std::ifstream in(path, std::ios::binary);
    if (!in) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(1); }

    in.seekg(0, std::ios::end);
    const std::streamoff bytes = in.tellg();
    in.seekg(0, std::ios::beg);

    out.resize((size_t)bytes / 4);
    in.read((char*)out.data(), (std::streamsize)(out.size() * 4));
    return out;
}

// Pack a list of strings into one buffer plus offsets, and hand both to the device.
struct Packed
{
    char*     text   = nullptr;
    uint32_t* offset = nullptr;
    uint32_t  count  = 0;
    size_t    bytes  = 0;
};

static Packed Upload(const std::vector<std::string>& items)
{
    Packed out;
    out.count = (uint32_t)items.size();

    std::vector<uint32_t> offset(items.size() + 1, 0);
    size_t total = 0;
    for (size_t i = 0; i < items.size(); i++) { offset[i] = (uint32_t)total; total += items[i].size(); }
    offset[items.size()] = (uint32_t)total;
    out.bytes = total;

    std::vector<char> text(total ? total : 1);
    for (size_t i = 0; i < items.size(); i++)
        std::memcpy(text.data() + offset[i], items[i].data(), items[i].size());

    CUDA_CHECK(cudaMalloc(&out.text, text.size()));
    CUDA_CHECK(cudaMalloc(&out.offset, offset.size() * 4));
    CUDA_CHECK(cudaMemcpy(out.text, text.data(), text.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out.offset, offset.data(), offset.size() * 4, cudaMemcpyHostToDevice));
    return out;
}

int main(int argc, char** argv)
{
    std::string prefixPath, suffixPath, targetPath, extendPath, outPath;
    std::string contOffPath, contIdxPath;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        const bool next = i + 1 < argc;
        if      (a == "--prefixes" && next) prefixPath = argv[++i];
        else if (a == "--suffixes" && next) suffixPath = argv[++i];
        else if (a == "--targets"  && next) targetPath = argv[++i];
        else if (a == "--extend"   && next) extendPath = argv[++i];
        else if (a == "--cont-offset" && next) contOffPath = argv[++i];
        else if (a == "--cont-index"  && next) contIdxPath = argv[++i];
        else if (a == "--out"      && next) outPath    = argv[++i];
    }
    if (prefixPath.empty() || suffixPath.empty() || targetPath.empty()) {
        std::fprintf(stderr, "usage: mitm_frag --prefixes F --suffixes F --targets F "
                             "[--extend F --cont-offset F --cont-index F] [--out F]\n");
        return 2;
    }

    const std::vector<std::string> prefixes = ReadLines(prefixPath);
    const std::vector<std::string> suffixes = ReadLines(suffixPath);
    const std::vector<std::string> words    = extendPath.empty()
                                            ? std::vector<std::string>() : ReadLines(extendPath);

    std::vector<uint64_t> targets;
    for (const std::string& line : ReadLines(targetPath)) {
        std::string hex = line;
        if (hex.rfind("hash_", 0) == 0) hex = hex.substr(5);
        if (hex.rfind("0x", 0) == 0)    hex = hex.substr(2);
        try { targets.push_back(std::stoull(hex, nullptr, 16) & MASK60); } catch (...) {}
    }
    if (prefixes.empty() || suffixes.empty() || targets.empty()) {
        std::fprintf(stderr, "empty input\n");
        return 2;
    }

    std::vector<uint32_t> contOff = ReadU32(contOffPath);
    std::vector<uint32_t> contIdx = ReadU32(contIdxPath);
    if (!contOff.empty() && contOff.size() != prefixes.size() + 1) {
        std::fprintf(stderr, "continuation offsets do not match the prefix list\n");
        return 2;
    }

    const uint64_t entries = contOff.empty() ? prefixes.size()
                                             : prefixes.size() + contIdx.size();
    const double pairs = (double)entries * (double)suffixes.size();

    std::printf("%zu prefixes", prefixes.size());
    if (!contOff.empty())
        std::printf(" + %zu continuations (%.0f each) = %llu entries",
                    contIdx.size(), (double)contIdx.size() / (double)prefixes.size(),
                    (unsigned long long)entries);
    std::printf(", %zu suffixes, %zu targets\n", suffixes.size(), targets.size());
    std::printf("%.3g pairs, %.1f expected false matches over the whole run\n",
                pairs, pairs * (double)targets.size() / 1.152921504606847e18);

    uint32_t slots = 1;
    while ((uint64_t)slots < entries * 2 && slots < (1u << 31)) slots <<= 1;

    const Packed dPrefix = Upload(prefixes);
    const Packed dWord   = Upload(words);
    const Packed dSuffix = Upload(suffixes);

    uint32_t *dContOff = nullptr, *dContIdx = nullptr;
    if (!contOff.empty()) {
        CUDA_CHECK(cudaMalloc(&dContOff, contOff.size() * 4));
        CUDA_CHECK(cudaMalloc(&dContIdx, contIdx.size() * 4));
        CUDA_CHECK(cudaMemcpy(dContOff, contOff.data(), contOff.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dContIdx, contIdx.data(), contIdx.size() * 4, cudaMemcpyHostToDevice));
    }

    uint64_t *dKeys = nullptr, *dVals = nullptr, *dTargets = nullptr, *dHits = nullptr;
    uint32_t *dHitCount = nullptr;
    const uint32_t hitMax = 1u << 21;

    CUDA_CHECK(cudaMalloc(&dKeys, (size_t)slots * 8));
    CUDA_CHECK(cudaMalloc(&dVals, (size_t)slots * 8));
    CUDA_CHECK(cudaMalloc(&dTargets, targets.size() * 8));
    CUDA_CHECK(cudaMalloc(&dHits, (size_t)hitMax * 16));
    CUDA_CHECK(cudaMalloc(&dHitCount, 4));
    CUDA_CHECK(cudaMemset(dKeys, 0xFF, (size_t)slots * 8));
    CUDA_CHECK(cudaMemset(dHitCount, 0, 4));
    CUDA_CHECK(cudaMemcpy(dTargets, targets.data(), targets.size() * 8, cudaMemcpyHostToDevice));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int grid = prop.multiProcessorCount * 16;
    std::printf("device %s, table %u slots (%.1f GB), suffix text %.0f MB\n",
                prop.name, slots, slots * 16.0 / 1e9, dSuffix.bytes / 1e6);

    auto started = std::chrono::steady_clock::now();
    Build<<<grid, 256>>>(dPrefix.text, dPrefix.offset, dPrefix.count,
                         dWord.text, dWord.offset, dContOff, dContIdx,
                         dKeys, dVals, slots - 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("table built in %.1f s\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());

    started = std::chrono::steady_clock::now();
    Search<<<grid, 256>>>(dSuffix.text, dSuffix.offset, dSuffix.count,
                          dTargets, (uint32_t)targets.size(),
                          dKeys, dVals, slots - 1, dHits, dHitCount, hitMax);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();

    uint32_t found = 0;
    CUDA_CHECK(cudaMemcpy(&found, dHitCount, 4, cudaMemcpyDeviceToHost));
    const uint32_t kept = std::min(found, hitMax);

    std::vector<uint64_t> hits((size_t)kept * 2);
    if (kept) CUDA_CHECK(cudaMemcpy(hits.data(), dHits, (size_t)kept * 16, cudaMemcpyDeviceToHost));

    std::printf("%.1f s, %.3g pairs/s, %u match(es)%s\n",
                secs, pairs / secs, found, found > hitMax ? " (buffer full)" : "");

    std::ofstream out;
    if (!outPath.empty()) out.open(outPath);
    for (uint32_t i = 0; i < kept; i++) {
        const uint64_t id = hits[i * 2];
        const uint64_t p  = id >> 24;
        const uint64_t k  = id & 0xFFFFFFull;

        std::string name = prefixes[(size_t)p];
        if (k) name += words[contIdx[contOff[(size_t)p] + (size_t)k - 1]] + "_";
        name += suffixes[(size_t)hits[i * 2 + 1]];

        if (out) out << name << "\n";
        else     std::printf("%s\n", name.c_str());
    }
    if (out) std::printf("written to %s\n", outPath.c_str());

    return 0;
}
