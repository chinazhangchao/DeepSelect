#include <cstdio>
#include <cstring>

#include <kerutils/kerutils.cuh>

static_assert(sizeof(longlong2) == 16 && sizeof(longlong2::x) == 8);

__global__ void shared_payload_round_trip(ku::uint128_t* output, float4* floats, ku::uint128_t input) {
    __shared__ ku::uint128_t value;
    if (threadIdx.x == 0) {
        ku::st_shared(&value, input);
        output[0] = ku::ld_shared(&value);
        ku::st_shared(&value, floats[0]);
        floats[1] = ku::ld_shared_float4(&value);
    }
}

__global__ void async_payload_store(ku::uint128_t input) {
    __shared__ ku::uint128_t value;
    __shared__ CUTE_ALIGNAS(16) char barrier_storage[sizeof(ku::transac_bar_t)];
    auto& barrier = *reinterpret_cast<ku::transac_bar_t*>(barrier_storage);
    if (threadIdx.x == 0) {
        barrier.init(1);
        cutlass::arch::fence_barrier_init();
        barrier.arrive_and_expect_tx(16);
        ku::st_async(&value, input, barrier);
        barrier.wait(0);
    }
}

bool check_payload(ku::uint128_t payload, uint64_t low, uint64_t high) {
    uint64_t words[2];
    std::memcpy(words, &payload, sizeof(payload));
    if (words[0] != low || words[1] != high) {
        std::fprintf(stderr, "128-bit payload did not preserve both 64-bit words\n");
        return false;
    }
    return true;
}

int main() {
    constexpr uint64_t cases[][2] = {
        {0, 0},
        {0x0123456789abcdefULL, 0xfedcba9876543210ULL},
        {0xffffffffffffffffULL, 0xffffffffffffffffULL},
        {0xff800000ff800000ULL, 0xff800000ff800000ULL},  // FP32 -INF
        {0xff80ff80ff80ff80ULL, 0xff80ff80ff80ff80ULL},  // BF16 -INF
    };
    if (!check_payload(ku::uint128_t{}, 0, 0)) return 1;
    for (const auto& words : cases) {
        if (!check_payload(ku::make_uint128(words[0], words[1]), words[0], words[1])) return 1;
    }
    constexpr auto filler = ku::make_uint128(cases[3][0], cases[3][1]);
    return check_payload(filler, cases[3][0], cases[3][1]) ? 0 : 1;
}
