package hgtmgr

import (
    "fmt"
    "log"
    "os"
    "math"
    "errors"
    "container/list"
    "vshed/cuhgt"
    "vshed/latlon"
    . "vshed/conf"
)

type cacheRecord struct {
    slot int
    users int
    ptr uint64
    ll latlon.LLi
    le *list.Element
}

type request struct {
    latlon.Recti
    replyto chan *Grid
    mask []bool
}

type HgtMgr struct {
    hgtdir string
    cache list.List
    cacheMap map[latlon.LLi]*cacheRecord
    rq chan request
    free chan *Grid
    opcount int
    cacheMiss int
    cacheRq int
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
        rq: make(chan request),
        free: make(chan *Grid),
        cacheMap: make(map[latlon.LLi]*cacheRecord),
    }

    for i:=0; i < HGTSLOTS; i++ {
        m.cache.PushFront(&cacheRecord{slot: i})
        m.cache.Front().Value.(*cacheRecord).le = m.cache.Front()
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
                    m.cacheRq++
                    cr, ok := m.cacheMap[ll]
                    if !ok {
                        m.cacheMiss++
                        for le := m.cache.Back(); le != nil; le = le.Prev() {
                            if le.Value.(*cacheRecord).users > 0 {
                                continue
                            }
                            cr = le.Value.(*cacheRecord)
                            ptr := cuhgt.Fetch(ll, m.hgtdir, cr.slot)
                            if ptr == 0 {
                                m.cacheMap[ll] = nil
                                g.mask[len(g.Map)] = false;
                                g.Map = append(g.Map, 0)
                                return
                            } else {
                                delete(m.cacheMap, cr.ll)
                                cr.ll = ll
                                cr.ptr = ptr
                                m.cacheMap[ll] = cr
                            }
                            break
                        }
                        if cr == nil {
                            log.Fatal("cache overflow!")
                        }
                    }
                    if cr == nil {
                        g.mask[len(g.Map)] = false;
                        g.Map = append(g.Map, 0)
                        return
                    }
                    cr.users++
                    g.Map = append(g.Map, cr.ptr)
                    m.cache.MoveToFront(cr.le)
                })
                /*for k, v := range m.cache {
                    fmt.Println(k, *v)
                }*/
                rq.replyto <- g
                m.opcount++
if m.opcount & 63 == 0 {
    fmt.Printf("Cache HIT: %.1f%%\n", float64(m.cacheRq - m.cacheMiss) * 100 / float64(m.cacheRq))
}
            case g := <-m.free:
                g.mgr = nil
                i := -1
                g.Recti.Apply(func (ll latlon.LLi) {
                    i++
                    if g.mask[i] == false {
                        return
                    }
                    cr, ok := m.cacheMap[ll]
                    if ok {
                        cr.users--
                    } else {
                        log.Println("Tile to free not in cache!", ll, g)
                    }
                })
        }
    }
}

func clamp(v, min, max float64) float64 {
    return math.Min(math.Max(v, min), max)
}
