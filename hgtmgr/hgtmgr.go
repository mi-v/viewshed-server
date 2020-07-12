package hgtmgr

import (
    //"fmt"
    "os"
    "errors"
    "vshed/cuhgt"
    "vshed/latlon"
)

type cacheRecord struct {
    score float64
    ptr uint64
    users int
}

type request struct {
    latlon.Recti
    replyto chan Grid
    mask []bool
}

type HgtMgr struct {
    hgtdir string
    cache map[latlon.LLi]*cacheRecord
    rq chan request
    free chan Grid
}

type Grid struct {
    Map []uint64
    latlon.Recti
    mask []bool
}

func New(dir string) (m *HgtMgr, err error) {
    fi, err := os.Stat(dir)
    if err != nil {
        return
    }

    if !fi.IsDir() {
        err = errors.New("not a dir: " + dir)
        return
    }

    m = &HgtMgr{
        hgtdir: dir,
        cache: make(map[latlon.LLi]*cacheRecord),
        rq: make(chan request),
        free: make(chan Grid),
    }
    go m.run()

    return
}

func (m *HgtMgr) GetGrid(rect latlon.Recti, mask []bool) Grid {
    r := make(chan Grid)
    m.rq <- request{Recti: rect, replyto: r, mask: mask}
    return <-r
}

func (m *HgtMgr) FreeGrid(g Grid) {
    m.free <- g
}

func (m *HgtMgr) Query(ll latlon.LL) float64 {
    g := m.GetGrid(latlon.Recti{LLi: ll.Floor().Int(), Width: 1, Height: 1}, []bool{true})
    hgt := cuhgt.Query(g.Map[0], ll)
    m.FreeGrid(g)
    return hgt
}

func (m *HgtMgr) run() {
    for {
        select {
            case rq := <-m.rq:
                g := Grid{
                    Map: make([]uint64, 0, rq.Width * rq.Height),
                    Recti: rq.Recti,
                }
                g.Recti.Apply(func (ll latlon.LLi) {
                    if rq.mask[len(g.Map)] {
                        cr, ok := m.cache[ll]
                        if !ok {
                            cr = &cacheRecord{ptr: cuhgt.Open(ll, m.hgtdir)}
                            m.cache[ll] = cr
                        }
                        cr.score++
                        cr.users++
                        g.Map = append(g.Map, cr.ptr)
                    } else {
                        g.Map = append(g.Map, 0)
                    }
                })
                rq.replyto <- g
            case g := <-m.free:
                g.Recti.Apply(func (ll latlon.LLi) {
                    m.cache[ll].users--
                })
                print("TODO: кеш доделать\n")
        }
    }
}
