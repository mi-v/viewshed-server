package deps

import (
    "vshed/tiler"
    "vshed/latlon"
    "vshed/metrics"
    "sync"
    "net/http"
)

type HgtGrid interface {
    PtrMap() []uint64
    Rect() latlon.Recti
    EvtReady() uint64
    Free()
}

type HgtMgr interface {
    GetGridAround(latlon.LL) HgtGrid
}

type Tiler interface {
    Encode(te tiler.Tile, dir string, report *sync.WaitGroup)
}

type MetricsCollector interface {
    Add(*metrics.Unit, int)
    Count(*metrics.Unit)
    Register(string, interface{})
    ServeHTTP(http.ResponseWriter, *http.Request)
}

type Container struct {
    HgtMgr HgtMgr
    Tiler Tiler
    MetricsCollector MetricsCollector
}
