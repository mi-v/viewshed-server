// not likely to change
#define PI 3.14159265359f
#define ERAD 6371000.0f
#define TORAD (PI / 180.0f)
#define MRDLEN (PI * ERAD)
#define CSLAT (MRDLEN / 180.0f) // cell size along latitude

#define DSTEPS 2048
#define ANGSTEPS 4096

typedef struct {
    float CUTOFF;
    float CUTON;
    int MAXZOOM;
} Config;

typedef struct cuErrX {
    const char* msg;
    int line;
} cuErrX;

typedef struct LLi {
    int lat, lon;
    #ifdef __cplusplus
    __host__ __device__ LLi operator+ (LLi b) {return LLi{lat + b.lat, lon + b.lon};};
    __host__ __device__ LLi operator- (LLi b) {return LLi{lat - b.lat, lon - b.lon};};
    #endif
} LLi;

typedef struct {LLi ll; int width, height;} Recti;

struct Vec3;
struct Px2;

typedef struct LL {
    float lat, lon;
    #ifdef __cplusplus
    __host__ __device__ LL toRad() {return LL{lat * TORAD, lon * TORAD};};
    __host__ __device__ LL fromRad() {return LL{lat / TORAD, lon / TORAD};};
    __host__ __device__ LL floor() {return LL{floorf(lat), floorf(lon)};};
    __host__ __device__ LL operator+ (LL b) {return LL{lat + b.lat, lon + b.lon};};
    __host__ __device__ LL operator- (LL b) {return LL{lat - b.lat, lon - b.lon};};
    __host__ __device__ LL operator/ (float b) {return LL{lat / b, lon / b};};
    __host__ __device__ LL operator+= (LL b) {lat+=b.lat; lon+=b.lon; return *this;};
    __host__ __device__ LL operator-= (LL b) {lat-=b.lat; lon-=b.lon; return *this;};
    __host__ __device__ operator LLi() {return LLi{int(floorf(lat)), int(floorf(lon))};};
    __host__ __device__ operator Vec3();
    __host__ __device__ Px2 toPx2(int z);
    #endif
} LL;

struct Vec3 {
    float x, y, z;
    #ifdef __cplusplus
    __host__ __device__ Vec3 operator+ (Vec3 b) {return Vec3{x+b.x, y+b.y, z+b.z};};
    __host__ __device__ Vec3 operator- (Vec3 b) {return Vec3{x-b.x, y-b.y, z-b.z};};
    __host__ __device__ float operator* (Vec3 b) {return x*b.x + y*b.y + z*b.z;};
    __host__ __device__ friend Vec3 operator* (float a, Vec3 b) {return Vec3{a*b.x, a*b.y, a*b.z};};
    __host__ __device__ operator float() {return norm3df(x, y, z);};
    #endif
};

typedef struct Px2 {
    int x, y;
    #ifdef __cplusplus
    __host__ __device__ Px2 operator+ (Px2 b) {return Px2{x+b.x, y+b.y};};
    __host__ __device__ Px2 operator- (Px2 b) {return Px2{x-b.x, y-b.y};};
    __host__ __device__ operator float() {return hypotf(x, y);};
    __host__ __device__ LL toLL(int zoom) {
        return LL{
            2 * atanf(expf(PI * (1 - float(y) / (128 << zoom)))) - PI / 2,
            (2 * PI * x) / (256 << zoom) - PI
        };
    };
    #endif
} Px2;

typedef struct {Px2 P; Px2 Q;} PxRect;

typedef struct Image {
    void* buf;
    PxRect rect;
    cuErrX error;
    #ifdef __cplusplus
        int w() {return rect.Q.x - rect.P.x;};
        int h() {return rect.Q.y - rect.P.y;};
        size_t wh() {return w() * h();};
    #endif
} Image;

#ifdef __cplusplus
__host__ __device__ LL::operator Vec3() {
    return Vec3{
        sinf(lon)*cosf(lat),
        sinf(lat),
        cosf(lon)*cosf(lat)
    };
};

__host__ __device__ Px2 LL::toPx2(int zoom) {
    return Px2{
        int((256 << zoom) * ((lon + PI) / (2 * PI))),
        int((128 << zoom) * (1 - logf(tanf(PI / 4 + lat / 2)) / PI))
    };
};
#endif
