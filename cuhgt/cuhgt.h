typedef struct cuErrX {
    int code;
    const char* msg;
    const char* file;
    int line;
} cuErrX;

typedef struct UploadResult {
    uint64_t ptr;
    cuErrX error;
} UploadResult;
