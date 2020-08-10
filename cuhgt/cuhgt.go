package cuhgt

import (
    "os"
    "fmt"
    _ "vshed/cuinit"
    "vshed/latlon"
    . "vshed/conf"
    "log"
)

// #include <stdint.h>
// #include <stdlib.h>
// #include "cuhgt.h"
// UploadResult uploadHgts(int fd, int slot);
// PrepResult prepareHgts(uint64_t *Hgts, int cnt);
// uint64_t cuhgtInit(int slots);
// #cgo LDFLAGS: -L../ -lcuhgt
import "C"

const hgtFileSize = 1201 * 1201 * 2
var hgtBase uint64

func Fetch(ll latlon.LLi, dir string, slot int) (ptr uint64) {
    hgtName := dir + "/" + mkHgtName(ll)
    hf, err := os.Open(hgtName)
    if err != nil {
        return
    }
    defer hf.Close()
    hfs, err := hf.Stat()
    if (err != nil || hfs.Size() != hgtFileSize) {
        return
    }

    cUR := C.uploadHgts(C.int(hf.Fd()), C.int(slot))
    if cUR.error.msg != nil {
        log.Fatalf("CUDA error: %d %s in %s:%d", cUR.error.code, C.GoString(cUR.error.msg), C.GoString(cUR.error.file), cUR.error.line)
        //log.Printf("CUDA error: %d %s in %s:%d", cUR.error.code, C.GoString(cUR.error.msg), C.GoString(cUR.error.file), cUR.error.line)
        return
    }
    return uint64(cUR.ptr)
}

func Prepare(prepq []uint64) (eptr uint64) {
    var cPR C.PrepResult
    if len(prepq) > 0 {
        cPR = C.prepareHgts((*C.ulong)(&prepq[0]), C.int(len(prepq)))
    } else {
        cPR = C.prepareHgts(nil, 0)
    }
    if cPR.error.msg != nil {
        log.Fatalf("CUDA error: %d %s in %s:%d", cPR.error.code, C.GoString(cPR.error.msg), C.GoString(cPR.error.file), cPR.error.line)
    }
    return uint64(cPR.eptr)
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
    log.Println("cuhgt init")
    hgtBase = uint64(C.cuhgtInit(HGTSLOTS))
}
