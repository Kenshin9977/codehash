// t7sweep - compose plausible identifiers and check them against Black Ops 3 script hashes.
//
// Different problem from the other two searchers here, and the difference is the width. Black Ops
// 3 hashes script names with Treyarch's canonical hash, which is FNV-1a 32-bit: seed 0x4B9ACE2F,
// prime 0x1000193, lowercased, with one extra multiply at the end.
//
// Thirty-two bits changes the arithmetic in both directions.
//
// Against it: the hash stops being evidence once there are many targets. Expected false matches
// are candidates * targets / 2^32, so a hundred false matches allows 4.8e9 candidates against 90
// targets and only 1.5e7 against 28 400. Sweeping everything at once returns noise. The way to
// use this is a small batch of targets at a time - the hashes that appear most in the decompiled
// scripts, which is what tools/t7_context.py ranks.
//
// For it: the whole hash space fits in memory. 2^32 bits is 512 MB, so the membership test is an
// exact bitmap rather than a filter - no false positives from the structure at all, only genuine
// collisions between two real strings. The sibling programs need a bloom filter and a sorted
// table behind it; here one bit lookup settles a candidate.
//
//   nvcc -O3 -arch=sm_86 -o t7sweep t7sweep.cu
//   t7sweep --targets top100.txt --prefixes prefixes.txt --dict words.txt --depth 2 --out found.txt
//
// Candidates are `<prefix>w1_w2_..._wD`, the prefix carrying its own trailing separator if it
// wants one. Cost is prefixes * words^depth, so depth is the setting to watch.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define T7_SEED  0x4B9ACE2Fu
#define T7_PRIME 0x1000193u

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "cuda: %s at line %d\n", cudaGetErrorString(e_), __LINE__); \
    std::exit(1); } } while (0)

// One dictionary entry is 32 bytes: up to 31 characters then the length, so an entry is two
// aligned 16-byte loads at an address every thread in the block shares.
#define WORD_STRIDE 32
#define WORD_MAX    31

struct Word { uint4 a, b; };

__device__ __forceinline__ uint32_t Step(uint32_t h, uint32_t c)
{
    return (h ^ c) * T7_PRIME;
}

// Hash the bytes of one dictionary word into a running state. Reads the 32-byte entry as two
// 16-byte vectors, which is what the stride is for.
__device__ __forceinline__ uint32_t HashWord(uint32_t h, Word w, uint32_t n)
{
    const uint32_t* p = reinterpret_cast<const uint32_t*>(&w);
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t byte = (p[i >> 2] >> ((i & 3) * 8)) & 0xFF;
        h = Step(h, byte);
    }
    return h;
}

__device__ __forceinline__ bool Marked(const uint32_t* __restrict__ bitmap, uint32_t h)
{
    return (bitmap[h >> 5] >> (h & 31)) & 1u;
}

// One thread owns a stem and walks the whole dictionary as leaves.
//
// The stem is hashed once; a candidate then costs only its final word, six or seven bytes rather
// than the twenty or thirty a whole identifier runs to. Every thread is on the same leaf at the
// same time, so the dictionary read is one broadcast per warp and no warp diverges on word
// length. Misses write nothing at all.
__global__ __launch_bounds__(256)
void Sweep(const char* __restrict__ preText, const uint32_t* __restrict__ preOff, uint32_t preCount,
           const Word* __restrict__ dict, uint32_t dictCount, uint32_t depth,
           const uint32_t* __restrict__ bitmap,
           uint64_t stemBegin, uint64_t stemEnd,
           uint64_t* __restrict__ hits, uint32_t* __restrict__ hitCount, uint32_t hitMax)
{
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

    for (uint64_t stem = stemBegin + blockIdx.x * blockDim.x + threadIdx.x;
         stem < stemEnd; stem += stride) {

        // Decode the stem: a prefix, then depth-1 words with a separator after each.
        uint64_t x = stem;
        uint32_t h = T7_SEED;

        for (uint32_t k = 1; k < depth; k++) {
            const uint32_t idx = (uint32_t)(x % dictCount);
            x /= dictCount;
            const Word w = dict[idx];
            h = HashWord(h, w, (w.b.w >> 24) & 0xFF);
            h = Step(h, '_');
        }

        // The prefix is hashed last in the decode but first in the string, so it has to lead.
        const uint32_t p = (uint32_t)(x % preCount);
        uint32_t base = T7_SEED;
        for (uint32_t i = preOff[p]; i < preOff[p + 1]; i++)
            base = Step(base, (uint32_t)(unsigned char)preText[i]);

        // Redo the words on top of the prefix state. Cheap next to the leaf sweep below, and it
        // keeps the decode above free of a second pass.
        x = stem;
        for (uint32_t k = 1; k < depth; k++) {
            const uint32_t idx = (uint32_t)(x % dictCount);
            x /= dictCount;
            const Word w = dict[idx];
            base = HashWord(base, w, (w.b.w >> 24) & 0xFF);
            base = Step(base, '_');
        }

        for (uint32_t j = 0; j < dictCount; j++) {
            const Word w = dict[j];
            const uint32_t len = (w.b.w >> 24) & 0xFF;
            const uint32_t full = HashWord(base, w, len) * T7_PRIME;   // the closing multiply

            if (Marked(bitmap, full)) {
                const uint32_t slot = atomicAdd(hitCount, 1u);
                if (slot < hitMax) hits[slot] = stem * (uint64_t)dictCount + j;
            }
        }
    }
}

static std::vector<std::string> ReadLines(const std::string& path, bool lower)
{
    std::vector<std::string> out;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line.empty()) continue;
        if (lower) for (char& c : line) c = (char)std::tolower((unsigned char)c);
        out.push_back(line);
    }
    return out;
}

static uint32_t HostHash(const std::string& s)
{
    uint32_t h = T7_SEED;
    for (unsigned char c : s) h = (h ^ (uint32_t)std::tolower(c)) * T7_PRIME;
    return h * T7_PRIME;
}

int main(int argc, char** argv)
{
    std::string targetPath, prefixPath, dictPath, outPath;
    uint32_t depth = 1;
    long long chunk = 1LL << 22;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        const bool next = i + 1 < argc;
        if      (a == "--targets"  && next) targetPath = argv[++i];
        else if (a == "--prefixes" && next) prefixPath = argv[++i];
        else if (a == "--dict"     && next) dictPath   = argv[++i];
        else if (a == "--out"      && next) outPath    = argv[++i];
        else if (a == "--depth"    && next) depth      = (uint32_t)std::atoi(argv[++i]);
        else if (a == "--chunk"    && next) chunk      = std::atoll(argv[++i]);
    }
    if (targetPath.empty() || dictPath.empty() || depth < 1) {
        std::fprintf(stderr, "usage: t7sweep --targets F --dict F [--prefixes F] "
                             "[--depth N] [--chunk N] [--out F]\n");
        return 2;
    }

    // No prefix list means one empty prefix: sweep bare word combinations. ReadLines drops blank
    // lines, so an "empty" file cannot express that and the option is simply optional.
    std::vector<std::string> prefixes;
    if (prefixPath.empty()) prefixes.push_back(std::string());
    else                    prefixes = ReadLines(prefixPath, true);
    std::vector<std::string> words = ReadLines(dictPath, true);

    std::vector<uint32_t> targets;
    for (const std::string& line : ReadLines(targetPath, true)) {
        std::string hex = line;
        if (hex.rfind("hash_", 0) == 0) hex = hex.substr(5);
        if (hex.rfind("0x", 0) == 0)    hex = hex.substr(2);
        try { targets.push_back((uint32_t)std::stoul(hex, nullptr, 16)); } catch (...) {}
    }

    size_t dropped = 0;
    words.erase(std::remove_if(words.begin(), words.end(),
                               [&](const std::string& w) {
                                   if (w.size() > WORD_MAX) { dropped++; return true; }
                                   return w.empty();
                               }), words.end());

    if (prefixes.empty() || words.empty() || targets.empty()) {
        std::fprintf(stderr, "empty input\n");
        return 2;
    }

    double candidates = (double)prefixes.size();
    for (uint32_t k = 0; k < depth; k++) candidates *= (double)words.size();

    std::printf("%zu prefixes, %zu words", prefixes.size(), words.size());
    if (dropped) std::printf(" (%zu dropped, over %d characters)", dropped, WORD_MAX);
    std::printf(", %zu targets, depth %u\n", targets.size(), depth);
    std::printf("%.3g candidates, %.2f expected false matches\n",
                candidates, candidates * (double)targets.size() / 4294967296.0);

    // The membership test is the whole 32-bit space, one bit per hash: 512 MB, no filter, no
    // collisions of its own. A hit is a real preimage of a real target.
    const size_t bitmapWords = (size_t)1 << 27;          // 2^32 bits
    std::vector<uint32_t> bitmap(bitmapWords, 0u);
    for (uint32_t t : targets) bitmap[t >> 5] |= 1u << (t & 31);

    std::vector<Word> packed(words.size());
    std::memset(packed.data(), 0, packed.size() * sizeof(Word));
    for (size_t i = 0; i < words.size(); i++) {
        std::memcpy(&packed[i], words[i].data(), words[i].size());
        reinterpret_cast<uint8_t*>(&packed[i])[WORD_STRIDE - 1] = (uint8_t)words[i].size();
    }

    std::vector<uint32_t> preOff(prefixes.size() + 1, 0);
    size_t total = 0;
    for (size_t i = 0; i < prefixes.size(); i++) { preOff[i] = (uint32_t)total; total += prefixes[i].size(); }
    preOff[prefixes.size()] = (uint32_t)total;

    std::vector<char> preText(total ? total : 1);
    for (size_t i = 0; i < prefixes.size(); i++)
        std::memcpy(preText.data() + preOff[i], prefixes[i].data(), prefixes[i].size());

    char* dPre = nullptr;
    uint32_t *dPreOff = nullptr, *dBitmap = nullptr, *dHitCount = nullptr;
    Word* dDict = nullptr;
    uint64_t* dHits = nullptr;
    const uint32_t hitMax = 1u << 20;

    CUDA_CHECK(cudaMalloc(&dPre, preText.size()));
    CUDA_CHECK(cudaMalloc(&dPreOff, preOff.size() * 4));
    CUDA_CHECK(cudaMalloc(&dDict, packed.size() * sizeof(Word)));
    CUDA_CHECK(cudaMalloc(&dBitmap, bitmapWords * 4));
    CUDA_CHECK(cudaMalloc(&dHits, (size_t)hitMax * 8));
    CUDA_CHECK(cudaMalloc(&dHitCount, 4));
    CUDA_CHECK(cudaMemcpy(dPre, preText.data(), preText.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPreOff, preOff.data(), preOff.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dDict, packed.data(), packed.size() * sizeof(Word), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBitmap, bitmap.data(), bitmapWords * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dHitCount, 0, 4));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int grid = prop.multiProcessorCount * 8;

    uint64_t stems = prefixes.size();
    for (uint32_t k = 1; k < depth; k++) stems *= words.size();

    std::printf("device %s, bitmap %.0f MB, %llu stems\n",
                prop.name, bitmapWords * 4 / 1e6, (unsigned long long)stems);

    const auto started = std::chrono::steady_clock::now();
    for (uint64_t begin = 0; begin < stems; begin += (uint64_t)chunk) {
        const uint64_t end = std::min(stems, begin + (uint64_t)chunk);
        Sweep<<<grid, 256>>>(dPre, dPreOff, (uint32_t)prefixes.size(),
                             dDict, (uint32_t)words.size(), depth,
                             dBitmap, begin, end, dHits, dHitCount, hitMax);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        const double done = (double)end / (double)stems;
        const double spent = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        std::printf("\r  %5.1f%%  %.3g cand/s  left %.0fs      ",
                    100.0 * done, candidates * done / spent,
                    spent > 0.5 ? spent * (1.0 - done) / done : 0.0);
        std::fflush(stdout);
    }

    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();

    uint32_t found = 0;
    CUDA_CHECK(cudaMemcpy(&found, dHitCount, 4, cudaMemcpyDeviceToHost));
    const uint32_t kept = std::min(found, hitMax);

    std::vector<uint64_t> hits(kept);
    if (kept) CUDA_CHECK(cudaMemcpy(hits.data(), dHits, (size_t)kept * 8, cudaMemcpyDeviceToHost));

    std::printf("\r  %.1f s, %.3g candidates/s, %u hit(s)%s\n",
                secs, candidates / secs, found, found > hitMax ? " (buffer full)" : "");

    std::ofstream out;
    if (!outPath.empty()) out.open(outPath);
    for (uint32_t i = 0; i < kept; i++) {
        uint64_t id = hits[i];
        const uint32_t leaf = (uint32_t)(id % words.size());
        id /= words.size();

        std::vector<uint32_t> mids;
        for (uint32_t k = 1; k < depth; k++) { mids.push_back((uint32_t)(id % words.size())); id /= words.size(); }

        std::string name = prefixes[(size_t)(id % prefixes.size())];
        for (uint32_t idx : mids) name += words[idx] + "_";
        name += words[leaf];

        // Recompute before reporting. The bitmap says a hash was wanted, not that this string is
        // what produced the one in the file, and a reconstruction bug looks exactly like a find.
        const uint32_t h = HostHash(name);
        if (out) out << std::hex << h << "," << name << "\n";
        else     std::printf("%08x,%s\n", h, name.c_str());
    }
    if (out) std::printf("written to %s\n", outPath.c_str());

    return 0;
}
