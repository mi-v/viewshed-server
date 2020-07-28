package hgtmgr

import (
    "fmt"
    "log"
    "os"
    "math"
    "sort"
    "errors"
    "vshed/cuhgt"
    "vshed/latlon"
    . "vshed/conf"
)

type cacheRecord struct {
    ptr uint64
    users int
    op int
    ll latlon.LLi
}

type request struct {
    latlon.Recti
    replyto chan *Grid
    mask []bool
}

type HgtMgr struct {
    hgtdir string
    cache map[latlon.LLi]*cacheRecord
    rq chan request
    free chan *Grid
    opcount int
}

type Grid struct {
    Map []uint64
    latlon.Recti
    mgr *HgtMgr
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
        free: make(chan *Grid),
    }
    go m.run()

    return
}

func (m *HgtMgr) GetGrid(rect latlon.Recti, mask []bool) *Grid {
    r := make(chan *Grid)
    m.rq <- request{Recti: rect, replyto: r, mask: mask}
    return <-r
}

func (m *HgtMgr) GetGridAround(ll latlon.LL) *Grid {
    r := latlon.RectiFromRadius(ll, CUTOFF / CSLAT + 0.1)
    mask := make([]bool, 0, r.Width * r.Height)
    r.Apply(func (cll latlon.LLi) {
        cutoff := CUTOFF / CSLAT + 0.1
        cllf := cll.Float()
        dY := clamp(ll.Lat, cllf.Lat, cllf.Lat + 1) - ll.Lat
        dX := (clamp(ll.Lon, cllf.Lon, cllf.Lon + 1) - ll.Lon) * math.Cos((ll.LatR() + cllf.LatR()) / 2)
        mask = append(mask, dX * dX + dY * dY < cutoff * cutoff)
    })

    return m.GetGrid(r, mask)
}

func (g *Grid) Free() {
    g.mgr.free <- g
}

func (m *HgtMgr) Query(ll latlon.LL) float64 {
    g := m.GetGrid(latlon.Recti{LLi: ll.Floor().Int(), Width: 1, Height: 1}, []bool{true})
    hgt := cuhgt.Query(g.Map[0], ll)
    g.Free()
    return hgt
}

func (m *HgtMgr) run() {
    for {
        select {
            case rq := <-m.rq:
                g := &Grid{
                    Map: make([]uint64, 0, rq.Width * rq.Height),
                    Recti: rq.Recti,
                    mgr: m,
                    mask: rq.mask,
                }
                g.Recti.Apply(func (ll latlon.LLi) {
                    if rq.mask[len(g.Map)] == false {
                        g.Map = append(g.Map, 0)
                        return
                    }
                    cr, ok := m.cache[ll]
                    if !ok {
                        ptr := cuhgt.Open(ll, m.hgtdir)
                        if ptr == 0 {
                            g.mask[len(g.Map)] = false;
                            g.Map = append(g.Map, 0)
                            return
                        }
                        cr = &cacheRecord{ptr: ptr, ll: ll}
                        m.cache[ll] = cr
                    }
                    cr.users++
                    cr.op = m.opcount
                    g.Map = append(g.Map, cr.ptr)
                })
                for k, v := range m.cache {
                    fmt.Println(k, *v)
                }
                rq.replyto <- g
                m.opcount++
                if len(m.cache) > HGTCACHECAP {
                    cch := make([]*cacheRecord, len(m.cache))
                    i := 0
                    for _, r := range(m.cache) {
                        cch[i] = r
                        i++
                    }
                    sort.Slice(cch, func (i, j int) bool {return cch[i].op < cch[j].op})
                    rm := len(cch) - HGTCACHECAP * 8 / 10
                    for _, r := range(cch) {
                        if r.users == 0 {
                            cuhgt.Free(r.ptr)
                            delete(m.cache, r.ll)
                            rm--
                            if rm == 0 {
                                break
                            }
                        }
                    }
                }
            case g := <-m.free:
                g.mgr = nil
                i := -1
                g.Recti.Apply(func (ll latlon.LLi) {
                    i++
                    if g.mask[i] == false {
                        return
                    }
                    cr, ok := m.cache[ll]
                    if ok {
                        cr.users--
                    } else {
                        log.Println("Tile to free not in cache!", ll, g)
                    }
                })
                print("TODO: кеш доделать\n")
        }
    }
}

func clamp(v, min, max float64) float64 {
    return math.Min(math.Max(v, min), max)
}
