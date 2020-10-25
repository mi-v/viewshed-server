#include <cuda_profiler_api.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include "cuvshed.h"

#define sindf(a) sinpif((a) / 180)
#define cosdf(a) cospif((a) / 180)

#define cuErr(call)  {cudaError_t err; if (cudaSuccess != (err=(call))) throw cuErrX{err, cudaGetErrorString(err), __FILE__, __LINE__};}
#define bswap32(i)  (((i & 0xFF) << 24) | ((i & 0xFF00) << 8) | ((i & 0xFF0000) >> 8) | ((i & 0xFF000000) >> 24))

enum visMapMode {
    VIS_IMAGE,
    VIS_TILES,
};

struct Context {
    cudaStream_t stream;
    const short** HgtMap;
    float* AzEleD;
    float* myAlt;
    unsigned char* TSbuf;
    unsigned char* TSbuf_h;
};

__constant__ float CUTON;
static Config config;

__device__ float interp(float a, float b, float f)
{
    return a + f * (b - a);
}

__device__ float seaDistR(LL p0, LL p1)
{
    LL d = p1 - p0;

    return 2 * asinf(sqrtf(
        sinf(d.lat/2) * sinf(d.lat/2) + cosf(p0.lat) * cosf(p1.lat) * sinf(d.lon/2) * sinf(d.lon/2)
    ));
}

__device__ float abElev(float a, float b, float d)
{
    a /= ERAD;
    b /= ERAD;
    if(0)return (a + d * sinf(0.83f * d/2) - b * cos(0.83f * d)) /
        (d * cos(0.83f * d/2) + b * sin(0.83f * d));
    return (a + d * sinf(d/2) - b * cos(d)) /
        (d * cos(d/2) + b * sin(d));
}

__device__ float hgtQuery(const short** __restrict__ HgtMap, Recti rect, LL ll)
{
    //LLi lli = ll;
    //LLi lliof = lli - rect.ll;
    LLi lliof = LLi(ll) - rect.ll;
    const short* hgtCell = HgtMap[lliof.lat * rect.width + lliof.lon];
    if (!hgtCell) return 0;

    ll -= ll.floor();

    float Y = ll.lat * 1200;
    float X = ll.lon * 1200;

    int Xi = floorf(X);
    int Yi = floorf(Y);

    float Xf = X - Xi;
    float Yf = Y - Yi;

    int ofs = 1200 * 1280 - Yi * 1280 + Xi;
    float a = hgtCell[ofs];
    float b = hgtCell[ofs + 1];
    float c = hgtCell[ofs - 1280];
    float d = hgtCell[ofs - 1280 + 1];

    return (a * (1 - Xf) + b * Xf) * (1 - Yf) + (c * (1 - Xf) + d * Xf) * Yf;
}

__global__ void Query(const short** __restrict__ HgtMap, Recti rect, LL ll, float* result) {
    if (blockIdx.x || blockIdx.y || threadIdx.x || threadIdx.y) return;
    *result = hgtQuery(HgtMap, rect, ll);
}

__global__ void altQuery(const short** __restrict__ HgtMap, Recti rect, LL ll, float myH, float* result) {
    if (blockIdx.x || blockIdx.y || threadIdx.x || threadIdx.y) return;
    *result = myH + hgtQuery(HgtMap, rect, ll);
}

__global__ void doScape(const short** __restrict__ HgtMap, Recti hgtRect, float* __restrict__ AzEleD, const float* __restrict__ myAlt, LL myL, float dstep)
{
    int az = blockIdx.x * blockDim.x + threadIdx.x;
    int distN = blockIdx.y * blockDim.y + threadIdx.y;
    float dist = CUTON + dstep * distN * (distN + 1) / 2;
    float rDist = dist / ERAD;

    float azR = 2 * PI * az / ANGSTEPS;

    LL myR = myL.toRad();
    LL ptR = {asinf(sindf(myL.lat) * cosf(rDist) + cosdf(myL.lat) * sinf(rDist) * cosf(azR))}; // <- lat only! lon follows
    ptR.lon = myR.lon + atan2f(sinf(azR) * sinf(rDist) * cosdf(myL.lat), cosf(rDist) - sindf(myL.lat) * sinf(ptR.lat));

    LL ptL = ptR.fromRad();

    float hgt = hgtQuery(HgtMap, hgtRect, ptL);

    float elev = abElev(*myAlt, hgt, rDist);

    int ofs = distN * ANGSTEPS + az;
    AzEleD[ofs] = elev;
}

__global__ void elevProject(float* AzEleD)
{
    int az = blockIdx.x * blockDim.x + threadIdx.x;
    float elev = AzEleD[ANGSTEPS + az];
    for (int distN = 1; distN < DSTEPS; distN++) {
        int ofs = distN * ANGSTEPS + az;
        if (AzEleD[ofs] > elev) {
            AzEleD[ofs] = elev;
        } else {
            elev = AzEleD[ofs];
        }
    }
}

template<visMapMode mode>
__global__ void doVisMap(
    const short** __restrict__ HgtMap,
    Recti hgtRect,
    const float* __restrict__ AzEleD,
    LL myL,
    const float* __restrict__ myAlt,
    float theirH,
    Px2 pxBase,
    unsigned char* __restrict__ visMap,
    int zoom,
    float dstep
)
{
    Px2 imgPx = {
        int(blockIdx.x * blockDim.x + threadIdx.x),
        int(blockIdx.y * blockDim.y + threadIdx.y)
    };
    int visMapWidth = blockDim.x * gridDim.x;

    Px2 ptPx = pxBase + imgPx;

    LL ptR = ptPx.toLL(zoom);
    LL ptL = ptR.fromRad();

    LL myR = myL.toRad();
    float distR = seaDistR(myR, ptR);

    float hgt = hgtQuery(HgtMap, hgtRect, ptL) + theirH;

    float elev = abElev(*myAlt, hgt, distR);

    float dist = ERAD * distR;
    int distN = floorf((sqrtf(1 + 8 * (dist - CUTON) / dstep) - 1) / 2);
    float distNdist = CUTON + dstep * distN * (distN + 1) / 2;

    float azR = atan2f(sinf(ptR.lon - myR.lon) * cosf(ptR.lat), cosf(myR.lat) * sinf(ptR.lat) - sinf(myR.lat) * cosf(ptR.lat) * cosf(ptR.lon - myR.lon));
    while (azR < 0) {
        azR += 2 * PI;
    }
    float azi;
    float azf = modff(ANGSTEPS * azR / (2 * PI), &azi);
    int az = azi;

    bool visible = false;

    if (dist < CUTON) {
        visible = true;
    }

    if (distN >= 0 && distN < DSTEPS && elev - 0.00005 <= interp(AzEleD[distN * ANGSTEPS + az % ANGSTEPS], AzEleD[distN * ANGSTEPS + (az+1) % ANGSTEPS], azf)) {
        Px2 myPx = myR.toPx2(zoom);

        float pxDist = float(ptPx - myPx);

        LL llStep = (ptL - myL) / pxDist;

        float distStep = dist / pxDist;
        float distRStep = distR / pxDist;

        visible = true;
        int i = 10;
        while (dist > distNdist && i--) {
            dist -= distStep;
            distR -= distRStep;
            ptL -= llStep;
            ptR = ptL.toRad();

            hgt = hgtQuery(HgtMap, hgtRect, ptL);

            float stepElev = abElev(*myAlt, hgt, distR);

            if (stepElev < elev) {
                visible = false;
                break;
            }
        }
    }

    int visMapOffset;
    if (mode == VIS_IMAGE) {
        visMapOffset = visMapWidth * imgPx.y + imgPx.x;
    } else {
        visMapOffset = ((visMapWidth / 256) * (imgPx.y / 256) + imgPx.x / 256) * 256 * 256 // tile start offset
            + 256 * (imgPx.y % 256) + (imgPx.x % 256); // offset inside the tile
    }

    unsigned bb = __brev(__ballot_sync(~0, visible));
    if (threadIdx.x % 8 == 0) {
        unsigned char b = bb >> (24 - threadIdx.x % 32); // warp is 32 threads, get the 8 bits we need into b
        visMap[visMapOffset / 8] = b;
    }
}

__global__ void unzoomTiles(uint32_t SrcTiles[][256][256/32], uint32_t DstTiles[][256][256/32], PxRect srcRect, PxRect dstRect, int zoom) {
    int dstTileIdx = int(blockIdx.y * blockDim.y + threadIdx.y) / 256;
    Px2 dstPosInTile = {
        int(blockIdx.x * blockDim.x + threadIdx.x) * 32, // each thread is 32 px wide
        int(blockIdx.y * blockDim.y + threadIdx.y) % 256
    };

    Px2 dstTilePos = dstRect[dstTileIdx];
    Px2 dstWorldPos = dstTilePos * 256 + dstPosInTile;
    Px2 srcWorldPos = dstWorldPos * 2;
    Px2 srcTilePos = srcWorldPos >> 8; // [impln dep] arithmetic shift to correctly round down negatives

    if (!srcRect.contains(srcTilePos)) {
        return;
    }

    Px2 srcPosInTile = srcWorldPos & 255;

    uint32_t out = 0;
    uint32_t in;
    uint32_t mask;

    in = SrcTiles [srcRect.indexOf(srcTilePos)] [srcPosInTile.y] [srcPosInTile.x / 32];
    in |= SrcTiles [srcRect.indexOf(srcTilePos)] [srcPosInTile.y+1] [srcPosInTile.x / 32];
    in |= in<<1;
    in = bswap32(in);
    mask = 1u<<31;

    for (int i = 0; i < 16; i++) {
        out |= in & mask;
        in <<= 1;
        mask >>= 1;
    }

    in = SrcTiles [srcRect.indexOf(srcTilePos)] [srcPosInTile.y] [srcPosInTile.x / 32 + 1];
    in |= SrcTiles [srcRect.indexOf(srcTilePos)] [srcPosInTile.y+1] [srcPosInTile.x / 32 + 1];
    in |= in>>1;
    in = bswap32(in);
    mask = 1;

    for (int i = 0; i < 16; i++) {
        out |= in & mask;
        in >>= 1;
        mask <<= 1;
    }

    if (zoom > 0) {
        DstTiles[dstTileIdx][dstPosInTile.y][blockIdx.x * blockDim.x + threadIdx.x] = bswap32(out);
    } else { // may wrap twice into the same tile
        atomicOr(&DstTiles[0][dstPosInTile.y][blockIdx.x * blockDim.x + threadIdx.x], bswap32(out));
    }
}

void inline clk(const char* str, cudaStream_t cus) {
    return;
    static clock_t t = 0;
    cudaStreamSynchronize(cus);
    if (str) printf("%s: %f\n", str, (float)(clock() - t) / CLOCKS_PER_SEC);
    t = clock();
}

extern "C" {
    float Query(uint64_t ictx, Recti rect, LL ll, float* buf) {
        Context* ctx = (Context*)ictx;
        cudaStream_t cus = ctx->stream;

        float result;
        Query<<<1, 1, 0, cus>>>(ctx->HgtMap, rect, ll, buf);
        cudaMemcpyAsync(&result, buf, sizeof(float), cudaMemcpyDeviceToHost, cus);
        return result;
    }

    TileStrip makeTileStrip(uint64_t ictx, LL myL, int myH, int theirH, float cutoff, const uint64_t* HgtMapIn, Recti hgtRect, uint64_t ihgtsReady) {
        Context* ctx = (Context*)ictx;
        const short** HgtMap_d = ctx->HgtMap;
        float* AzEleD_d = ctx->AzEleD;
        float* myAlt_d = ctx->myAlt;
        unsigned char* TSbuf_d = ctx->TSbuf;
        TileStrip TS = {nullptr};
        cudaStream_t cus = ctx->stream;
        cudaEvent_t *hgtsReady = reinterpret_cast<cudaEvent_t*>(ihgtsReady);
        float dstep = 2 * (cutoff - config.CUTON) / (DSTEPS * (DSTEPS - 1));

        try {
clk(nullptr, cus);
            cuErr(cudaMemcpyAsync(HgtMap_d, HgtMapIn, hgtRect.width * hgtRect.height * sizeof(uint64_t), cudaMemcpyHostToDevice, cus));

            cuErr(cudaEventSynchronize(*hgtsReady));
            cuErr(cudaEventDestroy(*hgtsReady));
            delete hgtsReady;

            altQuery<<<1, 1, 0, cus>>>(HgtMap_d, hgtRect, myL, myH, myAlt_d);

            doScape<<<dim3(ANGSTEPS/32, DSTEPS/32), dim3(32, 32), 0, cus>>>(
                HgtMap_d,
                hgtRect,
                AzEleD_d,
                myAlt_d,
                myL,
                dstep
            );
            cuErr(cudaGetLastError());

            elevProject<<<dim3(ANGSTEPS/256), dim3(256), 0, cus>>>(AzEleD_d);
            cuErr(cudaGetLastError());

            LL myR = myL.toRad();
            LL rngR = {cutoff / ERAD};
            rngR.lon = -rngR.lat / cosf(myR.lat);

            PxRect irect;

            int zoom = 0;
            while ((256 << zoom) < config.MAXWIDTH * PI / -rngR.lon) zoom++;
            zoom--;

            irect.P = (myR + rngR).toPx2(zoom);
            irect.P.x &= ~255;
            irect.P.y &= ~255;
            irect.Q = (myR - rngR).toPx2(zoom);
            irect.Q.x |= 255;
            irect.Q.y |= 255;
            irect.Q ++;
            irect = irect.cropY(256 << zoom);
//printf("Image: %d x %d, %d bytes, z: %d  lat=%f  lon=%f\n", irect.w(), irect.h(), (irect.wh() + 7) / 8, zoom, myL.lat, myL.lon);

            TS.setup(irect, zoom);
            TS.nbytes = (TS.z[0].pretiles + 1) * 256 * 256 / 8;

            doVisMap<VIS_TILES><<<dim3(irect.w()/32, irect.h()/32), dim3(32, 32), 0, cus>>>(
                HgtMap_d,
                hgtRect,
                AzEleD_d,
                myL,
                myAlt_d,
                theirH,
                irect.P,
                TSbuf_d,
                zoom,
                dstep
            );
            cuErr(cudaGetLastError());
clk("doVisMap", cus);

            for (int z = zoom; z > 0; z--) {
                // each thread will produce 32x1 pixels
                // so width of 8 will sweep the whole strip
                unzoomTiles<<<dim3(1, TS.z[z-1].ntiles * 256 / 128), dim3(8, 128), 0, cus>>>(
                    reinterpret_cast<uint32_t(*)[256][256/32]>(TSbuf_d) + TS.z[z].pretiles, // src tiles ptr cast to array[] of bit tiles of uint32 (hence the /32)
                    reinterpret_cast<uint32_t(*)[256][256/32]>(TSbuf_d) + TS.z[z-1].pretiles, // dst tiles ptr
                    TS.z[z].rect,
                    TS.z[z-1].rect,
                    z-1
                );
                cuErr(cudaGetLastError());
            }
            TS.z[0].ntiles = 1;
clk("unzoom", cus);

//printf("nby: %d\n", TS.nbytes);
            TS.buf = ctx->TSbuf_h;
            cuErr(cudaMemcpyAsync(TS.buf, TSbuf_d, TS.nbytes, cudaMemcpyDeviceToHost, cus));
            cudaStreamSynchronize(cus);
            cuErr(cudaMemsetAsync(TSbuf_d, 0, TS.nbytes, cus));
        } catch (cuErrX error) {
            TS.error = error;
        }

        return TS;
    }

    uint64_t makeContext() {
        Context* ctx = new Context{};
        try {
            cuErr(cudaStreamCreate(&ctx->stream));
            cuErr(cudaMalloc(&ctx->HgtMap, ceilf(2 * config.CUTOFF / CSLAT + 1) * ceilf(2 * config.CUTOFF / (CSLAT * cosdf(85.0f)) + 1) * sizeof(uint64_t)));
            cuErr(cudaMalloc(&ctx->AzEleD, ANGSTEPS * DSTEPS * sizeof(float)));
            cuErr(cudaMalloc(&ctx->myAlt, sizeof(float)));

            size_t TSbytes = 0;
            int width = config.MAXWIDTH;
            int height = (width * 11 + 9)/ 10;
            do {
                size_t zoombytes = 0;
                zoombytes = (width | 255) + 1 + 256; // rounded to tiles
                zoombytes *= (height | 255) + 1 + 256;
                zoombytes /= 8; // 1 bpp
                TSbytes += zoombytes;
                width = (width + 1) / 2;
                height = (height + 1) / 2;
            } while (height > 1);
//printf("nby: %d\n", TSbytes);
            cuErr(cudaMalloc(&ctx->TSbuf, TSbytes));
            cuErr(cudaMemsetAsync(ctx->TSbuf, 0, TSbytes, ctx->stream));
            cuErr(cudaHostAlloc(&ctx->TSbuf_h, TSbytes, 0));
        } catch (cuErrX error) {
            cudaStreamDestroy(ctx->stream);
            cudaFree(ctx->HgtMap);
            cudaFree(ctx->AzEleD);
            cudaFree(ctx->TSbuf);
            cudaFreeHost(ctx->TSbuf_h);
            ctx = nullptr;
        }
        return (uint64_t)ctx;
    }

    void cuvshedInit(Config c) {
        cudaMemcpyToSymbol(CUTON, &c.CUTON, sizeof(CUTON));
        config = c;
    }

    void stopprof() {
        cudaProfilerStop();
    }
}
