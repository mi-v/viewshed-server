package cuhgt

import (
    "os"
    "fmt"
    //"io/ioutil"
    //"unsafe"
    "vshed/latlon"
)

// #include <stdint.h>
// #include <stdlib.h>
// uint64_t upload(short* Hgt);
// float Query(uint64_t Hgt, float lat, float lon);
// #cgo LDFLAGS: -L../ -lcuhgt
import "C"

const hgtFileSize = 1201 * 1201 * 2

func Open(ll latlon.LLi, dir string) (ptr uint64) {
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

    //ptr = uintptr(C.upload(unsafe.Pointer(&hgt[0])))
    //ptr = uintptr(C.upload((*C.char)(&hgt[0])))
    ptr = uint64(C.upload((*C.short)(cbuf)))
    return
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
