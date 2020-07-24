#include <stdint.h>
#include <stdio.h>
//#include <time.h>

__global__ void byteswap(int32_t* Hgt) {
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int row = blockIdx.y*blockDim.y+threadIdx.y;

    if (row < 1201 && col < 602) {
        int ofs = row * 640 + col;
        Hgt[ofs] = ((Hgt[ofs] >> 8) & 0x00ff00ff) | ((Hgt[ofs] << 8) & 0xff00ff00);
    }
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
    short* upload(short* Hgt) {
        short* ptr;
clock_t t = clock();
        cudaMalloc((void**)&ptr, 1280 * 1201 * sizeof(short));
//printf("malloc: %f\n", (float)(clock() - t) / CLOCKS_PER_SEC);
//t = clock();
        cudaMemcpy2D(ptr, 2560, Hgt, 1201 * sizeof(short), 1201 * sizeof(short), 1201, cudaMemcpyHostToDevice);
//printf("memcpy: %f\n", (float)(clock() - t) / CLOCKS_PER_SEC);
        byteswap<<<dim3(19, 38), dim3(32, 32)>>>((int32_t*)ptr);
        return ptr;
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
