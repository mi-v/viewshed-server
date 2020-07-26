package cuvshed

import (
    . "vshed/conf"
    "vshed/latlon"
    "image"
    "vshed/img1b"
    "image/color"
    "log"
    //"time"
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
Image makeImage(LL ll, int, uint64_t* Hgt, Recti hgtRect);
TileStrip makeTileStrip(LL myL, int myH, const uint64_t* HgtMapIn, Recti hgtRect);
void stopprof();
*/
import "C"

func Image(ll latlon.LL, myH int, hgtmap []uint64, rect latlon.Recti) *img1b.Image {
    cimg := C.makeImage(
        C.LL{C.float(ll.Lat), C.float(ll.Lon)},
        C.int(myH),
        (*C.ulong)(&hgtmap[0]),
        C.Recti{
            C.LLi{C.int(rect.Lat), C.int(rect.Lon)},
            C.int(rect.Width),
            C.int(rect.Height),
        },
    )
    if (cimg.error.msg != nil) {
        log.Println("CUDA error:", C.GoString(cimg.error.msg), cimg.error.line)
        return nil
    }
    cr := cimg.rect;
    buf := C.GoBytes(cimg.buf, (cr.Q.x - cr.P.x) * (cr.Q.y - cr.P.y) / 8)
    C.free(cimg.buf)
    return &img1b.Image{
        Pix: buf,
        Stride: int(cr.Q.x - cr.P.x) / 8,
        Rect: image.Rect(int(cr.P.x), int(cr.P.y), int(cr.Q.x), int(cr.Q.y)),
        //Rect: image.Rect(0, 0, int(cr.width), int(cr.height)),
        Palette: color.Palette{
            color.Black,
            color.White,
        },
    }
}

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
    sz := make([]tiler.StripZoom, MAXZOOM + 1)
    for z:=0; z<=MAXZOOM; z++ {
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
        MAXZOOM: MAXZOOM,
    })
}
