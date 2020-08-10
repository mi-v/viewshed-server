package cuvshed

import (
    _ "vshed/cuinit"
    . "vshed/conf"
    "vshed/latlon"
    "fmt"
    "vshed/tiler"
    "errors"
    "log"
)

/*
#include <stdint.h>
#include <stdlib.h>
#include "cuvshed.h"
#cgo LDFLAGS: -L../ -lcuvshed

void cuvshedInit(Config c);
TileStrip makeTileStrip(uint64_t ictx, LL myL, int myH, int theirH, const uint64_t* HgtMapIn, Recti hgtRect, uint64_t ihgtsReady);
uint64_t makeContext();
void stopprof();
*/
import "C"

func TileStrip(ctx uint64, ll latlon.LL, myH int, theirH int, hgtmap []uint64, rect latlon.Recti, eventReady uint64) (*tiler.Strip, error) {
    cTS := C.makeTileStrip(
        C.ulong(ctx),
        C.LL{C.float(ll.Lat), C.float(ll.Lon)},
        C.int(myH),
        C.int(theirH),
        (*C.ulong)(&hgtmap[0]),
        C.Recti{
            C.LLi{C.int(rect.Lat), C.int(rect.Lon)},
            C.int(rect.Width),
            C.int(rect.Height),
        },
        C.ulong(eventReady),
    )
    if (cTS.error.msg != nil) {
        log.Fatalf("CUDA error: %d %s in %s:%d", cTS.error.code, C.GoString(cTS.error.msg), C.GoString(cTS.error.file), cTS.error.line)
        return nil, errors.New(fmt.Sprintf("CUDA error: %s in %s:%d", C.GoString(cTS.error.msg), C.GoString(cTS.error.file), cTS.error.line))
    }
    maxzoom := int(cTS.zoom);
    sz := make([]tiler.StripZoom, maxzoom + 1)
    for z:=0; z<=maxzoom; z++ {
        cz := cTS.z[z]
        sz[z] = tiler.StripZoom{
            Rect: tiler.Rect{
                tiler.Corner{int(cz.rect.P.x), int(cz.rect.P.y)},
                tiler.Corner{int(cz.rect.Q.x), int(cz.rect.Q.y)},
            },
            Ntiles: int(cz.ntiles),
            Pretiles: int(cz.pretiles),
        }
    }
    TS := tiler.NewStrip(
        (*[1<<30]byte)(cTS.buf)[:cTS.nbytes:cTS.nbytes],
        sz,
        nil,//func() {C.free(cTS.buf)},
    );
    //fmt.Println(cTS);
    return TS, nil
}

func MakeCtx() uint64 {
    ctx := C.makeContext();
    if ctx == 0 {
        log.Fatal("could not create context")
    }
    return uint64(ctx)
}

func init() {
    log.Println("cuvshed init")
    C.cuvshedInit(C.Config{
        CUTOFF: CUTOFF,
        CUTON: CUTON,
        MAXWIDTH: MAXWIDTH,
    })
}
