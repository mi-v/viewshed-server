package cuhgt

import (
    "os"
    "fmt"
    //"io/ioutil"
    //"unsafe"
    "vshed/latlon"
    "log"
    "time"
)

// #include <stdint.h>
// #include <stdlib.h>
// #include "cuhgt.h"
// UploadResult upload(short* Hgt);
// void freeHgt(uint64_t ptr);
// float Query(uint64_t Hgt, float lat, float lon);
// void Init();
// #cgo LDFLAGS: -L../ -lcuhgt
import "C"

const hgtFileSize = 1201 * 1201 * 2

func Open(ll latlon.LLi, dir string) (ptr uint64) {
t := time.Now()
    hgtName := dir + "/" + mkHgtName(ll)
    hf, err := os.Open(hgtName)
    if (err != nil) {
        return
    }
    defer hf.Close()
    hfs, err := hf.Stat()
    if (err != nil || hfs.Size() != hgtFileSize) {
        return
    }

    cbuf := C.malloc(hgtFileSize)
    defer C.free(cbuf)
    buf := ((*[hgtFileSize]byte)(cbuf))[:hgtFileSize:hgtFileSize]

    n, err := hf.Read(buf)
    if (err != nil || n != hgtFileSize) {
        return
    }
fmt.Println("Read: ", time.Since(t))

    cUR := C.upload((*C.short)(cbuf))
    if (cUR.error.msg != nil) {
        log.Fatalf("CUDA error: %d %s in %s:%d", cUR.error.code, C.GoString(cUR.error.msg), C.GoString(cUR.error.file), cUR.error.line)
        //log.Printf("CUDA error: %d %s in %s:%d", cUR.error.code, C.GoString(cUR.error.msg), C.GoString(cUR.error.file), cUR.error.line)
        return
    }
    return uint64(cUR.ptr)
}

func Free (ptr uint64) {
    C.freeHgt(C.ulong(ptr))
}

func Query (ptr uint64, ll latlon.LL) float64 {
    if ptr == 0 {
        return 0
    }
    return float64(C.Query(C.ulong(ptr), C.float(ll.Lat), C.float(ll.Lon)))
}

func mkHgtName(ll latlon.LLi) string {
    ns := 'N'
    if ll.Lat < 0 {
        ns = 'S'
        ll.Lat = -ll.Lat
    }

    ew := 'E'
    if ll.Lon < 0 {
        ew = 'W'
        ll.Lon = -ll.Lon
    }

    return fmt.Sprintf("%c%02d%c%03d.hgt", ns, ll.Lat, ew, ll.Lon)
}

func init() {
    C.Init()
}
