#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
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
void* Hgt_h;
short* Hgtbase;

void inline clk(const char* str = nullptr) {
    static clock_t t = 0;
    cudaStreamSynchronize(cus);
    if (str) printf("%s: %f\n", str, (float)(clock() - t) / CLOCKS_PER_SEC);
    t = clock();
}

extern "C" {
    UploadResult upload(int fd, int slot) {
        UploadResult Res = {0};
        short* Hgt_d = Hgtbase + slot * 1280 * 1201;
clk();
        try {
            read(fd, Hgt_h, HGTSIZE);
            cuErr(cudaMemcpy2DAsync(
                Hgt_d,
                1280 * sizeof(short),
                Hgt_h,
                1201 * sizeof(short),
                1201 * sizeof(short),
                1201,
                cudaMemcpyHostToDevice,
                cus
            ));
            //byteswap16<<<dim3(19, 38), dim3(32, 32), 0, cus>>>((int32_t*)Hgt_d);
            cuErr(cudaGetLastError());
            cuErr(cudaStreamSynchronize(cus));
            Res.ptr = (uint64_t)Hgt_d;
        } catch (cuErrX error) {
            Res.error = error;
        }
clk("Upload");
        return Res;
    }

    uint64_t cuhgtInit(int slots) {
        cudaStreamCreate(&cus);
        cudaHostAlloc(&Hgt_h, HGTSIZE, 0);

        cudaMalloc(&Hgtbase, slots * 1280 * 1201 * sizeof(short));
        return (uint64_t)Hgtbase;
    }
}
