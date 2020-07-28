package cuvshed

import (
    . "vshed/conf"
    "vshed/latlon"
    "fmt"
    "vshed/tiler"
    "errors"
)

/*
#include <stdint.h>
#include <stdlib.h>
#include "cuvshed.h"
#cgo LDFLAGS: -L../ -lcuvshed

void Init(Config c);
TileStrip makeTileStrip(LL myL, int myH, const uint64_t* HgtMapIn, Recti hgtRect);
void stopprof();
*/
import "C"

func TileStrip(ll latlon.LL, myH int, hgtmap []uint64, rect latlon.Recti) (*tiler.Strip, error) {
    cTS := C.makeTileStrip(
        C.LL{C.float(ll.Lat), C.float(ll.Lon)},
        C.int(myH),
        (*C.ulong)(&hgtmap[0]),
        C.Recti{
            C.LLi{C.int(rect.Lat), C.int(rect.Lon)},
            C.int(rect.Width),
            C.int(rect.Height),
        },
    )
    if (cTS.error.msg != nil) {
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
        func() {C.free(cTS.buf)},
    );
    //fmt.Println(cTS);
    return TS, nil
}

func init() {
    C.Init(C.Config{
        CUTOFF: CUTOFF,
        CUTON: CUTON,
        MAXWIDTH: MAXWIDTH,
    })
}
