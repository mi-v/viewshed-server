package deps

import (
    "vshed/hgtmgr"
    "vshed/tiler"
    "vshed/latlon"
    "vshed/metrics"
    "sync"
)

type HgtMgr interface {
    GetGridAround(latlon.LL) *hgtmgr.Grid
}

type Tiler interface {
    Encode(te tiler.Tile, dir string, report *sync.WaitGroup)
}

type MetricsCollector interface {
    Add(*metrics.Unit, int)
    Register(string, interface{})
}

type Container struct {
    HgtMgr HgtMgr
    Tiler Tiler
    MetricsCollector MetricsCollector
}
