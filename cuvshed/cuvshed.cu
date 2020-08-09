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

__constant__ float CUTOFF;
__constant__ float CUTON;
__constant__ float DSTEP;
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

__global__ void doScape(const short** __restrict__ HgtMap, Recti hgtRect, float* __restrict__ AzEleD, float myAlt, LL myL)
{
    int az = blockIdx.x * blockDim.x + threadIdx.x;
    int distN = blockIdx.y * blockDim.y + threadIdx.y;
    float dist = CUTON + DSTEP * distN * (distN + 1) / 2;
    float rDist = dist / ERAD;

    float azR = 2 * PI * az / ANGSTEPS;

    LL myR = myL.toRad();
    LL ptR = {asinf(sindf(myL.lat) * cosf(rDist) + cosdf(myL.lat) * sinf(rDist) * cosf(azR))}; // <- lat only! lon follows
    ptR.lon = myR.lon + atan2f(sinf(azR) * sinf(rDist) * cosdf(myL.lat), cosf(rDist) - sindf(myL.lat) * sinf(ptR.lat));

    LL ptL = ptR.fromRad();

    float hgt = hgtQuery(HgtMap, hgtRect, ptL);

    float elev = abElev(myAlt, hgt, rDist);

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
    float myAlt,
    float theirH,
    Px2 pxBase,
    unsigned char* __restrict__ visMap,
    int zoom
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

    float elev = abElev(myAlt, hgt, distR);

    float dist = ERAD * distR;
    int distN = floorf((sqrtf(1 + 8 * (dist - CUTON) / DSTEP) - 1) / 2);
    float distNdist = CUTON + DSTEP * distN * (distN + 1) / 2;

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

            float stepElev = abElev(myAlt, hgt, distR);

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

/*template __global__ void doVisMap<VIS_IMAGE>(
    const short** __restrict__ HgtMap,
    Recti hgtRect,
    const float* __restrict__ AzEleD,
    LL myL,
    float myAlt,
    Px2 pxBase,
    unsigned char* __restrict__ visMap
);

template __global__ void doVisMap<VIS_TILES>(
    const short** __restrict__ HgtMap,
    Recti hgtRect,
    const float* __restrict__ AzEleD,
    LL myL,
    float myAlt,
    Px2 pxBase,
    unsigned char* __restrict__ visMap
);*/

__global__ void unzoomTiles(uint32_t SrcTiles[][256][256/32], uint32_t DstTiles[][256][256/32], PxRect srcRect, PxRect dstRect) {
    int dstTileIdx = int(blockIdx.y * blockDim.y + threadIdx.y) / 256;
    Px2 dstPosInTile = {
        int(blockIdx.x * blockDim.x + threadIdx.x) * 32, // each thread is 32 px wide
        int(blockIdx.y * blockDim.y + threadIdx.y) % 256
    };

    Px2 dstTilePos = dstRect[dstTileIdx];
    Px2 dstWorldPos = dstTilePos * 256 + dstPosInTile;
    Px2 srcWorldPos = dstWorldPos * 2;
    Px2 srcTilePos = srcWorldPos / 256;

    if (!srcRect.contains(srcTilePos)) {
        DstTiles[dstTileIdx][dstPosInTile.y][blockIdx.x * blockDim.x + threadIdx.x] = 0;//x10001000;
        return;
    }

    Px2 srcPosInTile = srcWorldPos % 256;

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

    DstTiles[dstTileIdx][dstPosInTile.y][blockIdx.x * blockDim.x + threadIdx.x] = bswap32(out);
}

void inline clk(const char* str, cudaStream_t cus) {
    return;
    static clock_t t = 0;
    cudaStreamSynchronize(cus);
    if (str) printf("%s: %f\n", str, (float)(clock() - t) / CLOCKS_PER_SEC);
    t = clock();
}

extern "C" {
    float Query(const short** HgtMap, Recti rect, LL ll) {
        float* d_result;
        float result;
        cudaMalloc((void**)&d_result, sizeof(float));
        Query<<<1, 1>>>(HgtMap, rect, ll, d_result);
        int r = cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_result);
        return result;
    }

    TileStrip makeTileStrip(LL myL, int myH, int theirH, const uint64_t* HgtMapIn, Recti hgtRect) {
        const short** HgtMap_d = nullptr;
        float* AzEleD_d = nullptr;
        unsigned char* TSbuf_d = nullptr;
        TileStrip TS = {nullptr};
        cudaStream_t cus=0;

        cudaStreamCreate(&cus);

        try {
clk(nullptr, cus);
            cuErr(cudaMalloc(&HgtMap_d, hgtRect.width * hgtRect.height * sizeof(uint64_t)));
            cuErr(cudaMemcpyAsync(HgtMap_d, HgtMapIn, hgtRect.width * hgtRect.height * sizeof(uint64_t), cudaMemcpyHostToDevice, cus));

            float myAlt = Query(HgtMap_d, hgtRect, myL) + myH;
            LL myR = myL.toRad();

            cuErr(cudaMalloc(&AzEleD_d, ANGSTEPS * DSTEPS * sizeof(float)));

            doScape<<<dim3(ANGSTEPS/32, DSTEPS/32), dim3(32, 32), 0, cus>>>(
                HgtMap_d,
                hgtRect,
                AzEleD_d,
                myAlt,
                myL
            );
            cuErr(cudaGetLastError());

            elevProject<<<dim3(ANGSTEPS/256), dim3(256), 0, cus>>>(AzEleD_d);
            cuErr(cudaGetLastError());

            LL rngR = {config.CUTOFF / ERAD};
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
printf("Image: %d x %d, %d bytes, z: %d  lat=%f  lon=%f\n", irect.w(), irect.h(), (irect.wh() + 7) / 8, zoom, myL.lat, myL.lon);

            TS.setup(irect, zoom);
            TS.nbytes = (TS.z[0].pretiles + 1) * 256 * 256 / 8;

            cuErr(cudaMalloc(&TSbuf_d, TS.nbytes));

            doVisMap<VIS_TILES><<<dim3(irect.w()/32, irect.h()/32), dim3(32, 32), 0, cus>>>(
                HgtMap_d,
                hgtRect,
                AzEleD_d,
                myL,
                myAlt,
                theirH,
                irect.P,
                TSbuf_d,
                zoom
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
                    TS.z[z-1].rect
                );
                cuErr(cudaGetLastError());
            }
clk("unzoom", cus);

            TS.buf = malloc(TS.nbytes);
            cuErr(cudaMemcpyAsync(TS.buf, TSbuf_d, TS.nbytes, cudaMemcpyDeviceToHost, cus));
        } catch (cuErrX error) {
            free(TS.buf);
            TS.error = error;
        }

        cudaStreamSynchronize(cus);
        cudaStreamDestroy(cus);
        cudaFree(HgtMap_d);
        cudaFree(AzEleD_d);
        cudaFree(TSbuf_d);
        return TS;
    }

    void Init(Config c) {
        cudaMemcpyToSymbol(CUTOFF, &c.CUTOFF, sizeof(CUTOFF));
        cudaMemcpyToSymbol(CUTON, &c.CUTON, sizeof(CUTON));
        float dstep = 2 * (c.CUTOFF - c.CUTON) / (DSTEPS * (DSTEPS - 1));
        cudaMemcpyToSymbol(DSTEP, &dstep, sizeof(DSTEP));
        config = c;
    }

    void stopprof() {
        cudaProfilerStop();
    }
}
