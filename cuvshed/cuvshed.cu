#include <cuda_profiler_api.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include "cuvshed.h"

#define sindf(a) sinpif((a) / 180)
#define cosdf(a) cospif((a) / 180)

#define cuErr(call) {if (cudaSuccess != (call)) throw cuErrX{cudaGetErrorString(cudaGetLastError()), __LINE__};}

__constant__ float CUTOFF;
__constant__ float CUTON;
__constant__ int MAXZOOM;
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

__global__ void doScape(const short** __restrict__ HgtMap, Recti hgtRect, float* __restrict__ AzEleD, Vec3 myP, float myH, LL myL)
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

    Vec3 ptP = (ERAD+hgt) * Vec3(ptR);

    Vec3 losP = myP - ptP;

    float trueDist = float(losP);

    float elev = 1 - myP * losP / (trueDist * (ERAD+myH));

    int ofs = distN * ANGSTEPS + az;
    AzEleD[ofs] = elev;
}

__global__ void elevProject(float* AzEleD)
{
    int az = blockIdx.x * blockDim.x + threadIdx.x;
    float elev = -1;
    for (int distN = 1; distN < DSTEPS; distN++) {
        int ofs = distN * ANGSTEPS + az;
        if (AzEleD[ofs] < elev) {
            AzEleD[ofs] = elev;
        } else {
            elev = AzEleD[ofs];
        }
    }
}

__global__ void doVisMap(
    const short** __restrict__ HgtMap,
    Recti hgtRect,
    const float* __restrict__ AzEleD,
    Vec3 myP,
    float myH,
    LL myL,
    Px2 pxBase,
    unsigned char* __restrict__ visMap
)
{
    Px2 imgPx = {
        int(blockIdx.x * blockDim.x + threadIdx.x),
        int(blockIdx.y * blockDim.y + threadIdx.y)
    };
    int visMapWidth = blockDim.x * gridDim.x;

    Px2 ptPx = pxBase + imgPx;

    LL ptR = ptPx.toLL(MAXZOOM);

    LL ptL = ptR.fromRad();

    float hgt = hgtQuery(HgtMap, hgtRect, ptL);

    Vec3 ptP = (ERAD+hgt) * Vec3(ptR);
    Vec3 losP = myP - ptP;

    float trueDist = float(losP);

    float elev = 1 - myP * losP / (trueDist * (ERAD+myH));

    LL myR = myL.toRad();

    float dist = ERAD * seaDistR(myR, ptR);
    int distN = floorf((sqrtf(1 + 8 * (dist - CUTON) / DSTEP) - 1) / 2);
    float distNdist = CUTON + DSTEP * distN * (distN + 1) / 2;

    float azR = atan2f(sinf(ptR.lon - myR.lon) * cosf(ptR.lat), cosf(myR.lat) * sinf(ptR.lat) - sinf(myR.lat) * cosf(ptR.lat) * cosf(ptR.lon - myR.lon));
    if (azR < 0) {
        azR += 2 * PI;
    }
    float azi;
    float azf = modff(ANGSTEPS * azR / (2 * PI), &azi);
    int az = azi;

    //visMap[visMapOffset] = 0;

    //if (distN >= DSTEPS) {
    /*if (__all_sync(~0, distN >= DSTEPS)) {
        //visMap[visMapOffset] = 1;
        return;
    }*/
    bool visible = false;

    if (distN < DSTEPS && elev + 0.0001 > interp(AzEleD[distN * ANGSTEPS + az], AzEleD[distN * ANGSTEPS + (az+1) % ANGSTEPS], azf)) {
        Px2 myPx = myR.toPx2(MAXZOOM);

        float pxDist = float(ptPx - myPx);

        LL llStep = (ptL - myL) / pxDist;

        float distStep = dist / pxDist;

        visible = true;
        int i = 10;
        while (dist > distNdist && i--) {
            dist -= distStep;
            ptL -= llStep;
            ptR = ptL.toRad();

            hgt = hgtQuery(HgtMap, hgtRect, ptL);

            Vec3 ptP = (ERAD+hgt) * Vec3(ptR);
            Vec3 losP = myP - ptP;

            trueDist = float(losP);

            float stepElev = 1 - myP * losP / (trueDist * (ERAD+myH));

            if (stepElev > elev) {
                visible = false;
                break;
            }
        }

        //visMap[visMapOffset] = 1;
    }

    int visMapOffset = visMapWidth * imgPx.y + imgPx.x;
    //visMap[visMapOffset] = visible ? 1 : 0;

    //unsigned char b = __brev(__ballot_sync(~0, visible)) >> (threadIdx.x % 32 - 24);
    unsigned bb = __brev(__ballot_sync(~0, visible));
    if (threadIdx.x % 8 == 0) {
        unsigned char b = bb >> (24 - threadIdx.x % 32);
        visMap[visMapOffset / 8] = b;
    }
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

    Image makeImage(LL myL, int myH, const uint64_t* HgtMapIn, Recti hgtRect) {
        const short** HgtMap = NULL;
        float* AzEleD = NULL;
        unsigned char* Img_d = NULL;
        Image Img = {};

        try {
            cuErr(cudaMalloc(&HgtMap, hgtRect.width * hgtRect.height * sizeof(uint64_t)));
            cuErr(cudaMemcpy(HgtMap, HgtMapIn, hgtRect.width * hgtRect.height * sizeof(uint64_t), cudaMemcpyHostToDevice));

            LL myR = myL.toRad();
            Vec3 myP = (ERAD + Query(HgtMap, hgtRect, myL) + myH) * Vec3(myR);

            cuErr(cudaMalloc(&AzEleD, ANGSTEPS * DSTEPS * sizeof(float)));

            doScape<<<dim3(ANGSTEPS/256, DSTEPS), dim3(256, 1)>>>(
                HgtMap,
                hgtRect,
                AzEleD,
                myP,
                myH,
                myL
            );
            cuErr(cudaGetLastError());

            elevProject<<<dim3(ANGSTEPS/32), dim3(32)>>>(AzEleD);
            cuErr(cudaGetLastError());

            LL rngR = {config.CUTOFF / ERAD};
            rngR.lon = -rngR.lat / cosf(myR.lat);

            Img.rect.P = (myR + rngR).toPx2(config.MAXZOOM);
            Img.rect.P.x &= ~255;
            Img.rect.P.y &= ~255;
            Img.rect.Q = (myR - rngR).toPx2(config.MAXZOOM);
            Img.rect.Q.x |= 255;
            Img.rect.Q.y |= 255;
            Img.rect.Q.x += 1;
            Img.rect.Q.y += 1;

            cuErr(cudaMalloc(&Img_d, Img.wh() / 8));

            doVisMap<<<dim3(Img.w()/256, Img.h()), dim3(256, 1)>>>(
                HgtMap,
                hgtRect,
                AzEleD,
                myP,
                myH,
                myL,
                Img.rect.P,
                Img_d
            );
            cuErr(cudaGetLastError());

            Img.buf = malloc(Img.wh());
            cuErr(cudaMemcpy(Img.buf, Img_d, Img.wh() / 8, cudaMemcpyDeviceToHost));
        } catch (cuErrX error) {
            Img.error = error;
        }

        cudaFree(HgtMap);
        cudaFree(AzEleD);
        cudaFree(Img_d);
        return Img;
    }

    void Init(Config c) {
        cudaMemcpyToSymbol(CUTOFF, &c.CUTOFF, sizeof(CUTOFF));
        cudaMemcpyToSymbol(CUTON, &c.CUTON, sizeof(CUTON));
        cudaMemcpyToSymbol(MAXZOOM, &c.MAXZOOM, sizeof(MAXZOOM));
        float dstep = 2 * (c.CUTOFF - c.CUTON) / (DSTEPS * (DSTEPS - 1));
        cudaMemcpyToSymbol(DSTEP, &dstep, sizeof(DSTEP));
        config = c;
    }

    void stopprof() {
        cudaProfilerStop();
    }
}
