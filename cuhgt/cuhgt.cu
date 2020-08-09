#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <sys/mman.h>
#include <cerrno>
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

cudaStream_t cus=0;

void inline clk(const char* str = nullptr) {
    static clock_t t = 0;
    cudaStreamSynchronize(cus);
    if (str) printf("%s: %f\n", str, (float)(clock() - t) / CLOCKS_PER_SEC);
    t = clock();
}

__global__ void Query(const short* __restrict__ Hgt, float lat, float lon, float* result) {
    if (blockIdx.x || blockIdx.y || threadIdx.x || threadIdx.y) return;

    lat -= floorf(lat);
    lon -= floorf(lon);

    float Y = lat * 1200;
    float X = lon * 1200;

    int Xi = floorf(X);
    int Yi = floorf(Y);

    float Xf = X - Xi;
    float Yf = Y - Yi;

    int ofs = 1200 * 1280 - Yi * 1280 + Xi;
    float a = Hgt[ofs];
    float b = Hgt[ofs + 1];
    float c = Hgt[ofs - 1280];
    float d = Hgt[ofs - 1280 + 1];

    *result = (a * (1 - Xf) + b * Xf) * (1 - Yf) + (c * (1 - Xf) + d * Xf) * Yf;
}

extern "C" {
    UploadResult upload(int fd) {
        void* Hgt;
        void* Hgt_d;
        UploadResult Res = {0};
clk();
        try {
            Hgt = mmap(nullptr, HGTSIZE, PROT_READ, MAP_SHARED, fd, 0);
            cuErr(cudaMalloc((void**)&Hgt_d, 1280 * 1201 * sizeof(short)));
            cuErr(cudaMemcpy2DAsync(Hgt_d, 2560, Hgt, 1201 * sizeof(short), 1201 * sizeof(short), 1201, cudaMemcpyHostToDevice, cus));
            byteswap16<<<dim3(19, 38), dim3(32, 32), 0, cus>>>((int32_t*)Hgt_d);
            cuErr(cudaGetLastError());
            Res.ptr = (uint64_t)Hgt_d;
            cuErr(cudaStreamSynchronize(cus));
        } catch (cuErrX error) {
            Res.error = error;
        }
        if (Hgt != MAP_FAILED) {
            munmap(Hgt, HGTSIZE);
        }
clk("Upload");
        return Res;
    }

    void freeHgt(uint64_t Hgt_d) {
        cudaFree((void*)Hgt_d);
    }

    void Init() {
        cudaStreamCreate(&cus);
    }

    float Query(uint64_t Hgt, float lat, float lon) {
        float* d_result;
        float result;
        cudaMalloc((void**)&d_result, sizeof(float));
        Query<<<1, 1>>>((short*)Hgt, lat, lon, d_result);
        int r = cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_result);
        return result;
    }
}
