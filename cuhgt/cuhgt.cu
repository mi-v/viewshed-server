#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include "cuhgt.h"

#define cuErr(call)  {cudaError_t err; if (cudaSuccess != (err=(call))) throw cuErrX{err, cudaGetErrorString(err), __FILE__, __LINE__};}

__global__ void byteswap16(int32_t* Hgt) {
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int row = blockIdx.y*blockDim.y+threadIdx.y;

    if (row < 1201 && col < 602) {
        int ofs = row * 640 + col;
        Hgt[ofs] = ((Hgt[ofs] >> 8) & 0x00ff00ff) | ((Hgt[ofs] << 8) & 0xff00ff00);
    }
}

cudaStream_t cus;
void* Hgt_ha;
void* Hgt_hb;
short* Hgtbase;
cudaEvent_t *xfer_a = new cudaEvent_t;
cudaEvent_t *xfer_b = new cudaEvent_t;

void inline clk(const char* str = nullptr) {
    static clock_t t = 0;
    cudaStreamSynchronize(cus);
    if (str) printf("%s: %f\n", str, (float)(clock() - t) / CLOCKS_PER_SEC);
    t = clock();
}

extern "C" {
    UploadResult uploadHgts(int fd, int slot) {
        UploadResult Res = {0};
        short* Hgt_d = Hgtbase + slot * 1280 * 1201;
        try {
            cudaEventSynchronize(*xfer_a);
            read(fd, Hgt_ha, HGTSIZE);
            cuErr(cudaMemcpy2DAsync(
                Hgt_d,
                1280 * sizeof(short),
                Hgt_ha,
                1201 * sizeof(short),
                1201 * sizeof(short),
                1201,
                cudaMemcpyHostToDevice,
                cus
            ));
            cuErr(cudaEventRecord(*xfer_a, cus));
            std::swap(xfer_a, xfer_b);
            std::swap(Hgt_ha, Hgt_hb);
            Res.ptr = (uint64_t)Hgt_d;
        } catch (cuErrX error) {
            Res.error = error;
        }
        return Res;
    }

    PrepResult prepareHgts(uint64_t *Hgts, int cnt) {
        PrepResult Res = {0};
        cudaEvent_t *evt = new cudaEvent_t;
        try {
            for (int i = 0; i < cnt; i++) {
                byteswap16<<<dim3(19, 38), dim3(32, 32), 0, cus>>>(reinterpret_cast<int32_t*>(Hgts[i]));
            }
            cuErr(cudaGetLastError());
            //cuErr(cudaEventCreateWithFlags(evt, cudaEventBlockingSync | cudaEventDisableTiming));
            cuErr(cudaEventCreateWithFlags(evt, cudaEventDisableTiming));
            cuErr(cudaEventRecord(*evt, cus));
        } catch (cuErrX error) {
            Res.error = error;
        }
        Res.eptr = (uint64_t)evt;
        return Res;
    }

    uint64_t cuhgtInit(int slots) {
        cudaStreamCreate(&cus);
        cudaHostAlloc(&Hgt_ha, HGTSIZE, cudaHostAllocWriteCombined);
        cudaHostAlloc(&Hgt_hb, HGTSIZE, cudaHostAllocWriteCombined);
        cudaEventCreateWithFlags(xfer_a, cudaEventDisableTiming);
        cudaEventCreateWithFlags(xfer_b, cudaEventDisableTiming);

        cudaMalloc(&Hgtbase, slots * 1280 * 1201 * sizeof(short));
        return (uint64_t)Hgtbase;
    }
}
