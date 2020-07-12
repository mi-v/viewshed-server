package cuvshed

import (
    . "vshed/conf"
    "vshed/latlon"
    "image"
    "image/color"
    "log"
)

/*
#include <stdint.h>
#include <stdlib.h>
#include "cuvshed.h"
#cgo LDFLAGS: -L../ -lcuvshed

void Init(Config c);
Image makeImage(LL ll, int, uint64_t* Hgt, Recti rect);
void stopprof();
*/
import "C"

func Image(ll latlon.LL, myH int, hgtmap []uint64, rect latlon.Recti) image.Image {
    //cbuf := C.malloc(4096*2048)
//for i := 0; i < 100; i++ {
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
    if (cimg.error != nil) {
        log.Println("CUDA error:", C.GoString(cimg.error))
        return nil
    }
    cr := cimg.rect;
    buf := C.GoBytes(cimg.buf, (cr.Q.x - cr.P.x) * (cr.Q.y - cr.P.y))
    C.free(cimg.buf)
    return &image.Paletted{
        Pix: buf,
        Stride: int(cr.Q.x - cr.P.x),
        Rect: image.Rect(int(cr.P.x), int(cr.P.y), int(cr.Q.x), int(cr.Q.y)),
        //Rect: image.Rect(0, 0, int(cr.width), int(cr.height)),
        Palette: color.Palette{
            color.Gray{0},
            color.Gray{255},
        },
    }
//}
//C.stopprof()
    //return buf
}

func init() {
    C.Init(C.Config{
        CUTOFF: CUTOFF,
        CUTON: CUTON,
        MAXZOOM: MAXZOOM,
    })
}
