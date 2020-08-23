package hgtmgr

import (
    "log"
    "os"
    "math"
    "errors"
    "container/list"
    "vshed/deps"
    "vshed/metrics"
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
}

type Grid struct {
    ptrMap []uint64
    evtReady uint64
    rect latlon.Recti
    mgr *HgtMgr
    mask []bool
}

var mtx struct {
    CacheRequest metrics.Unit `mtx:". Cache requests"`
    CacheHit metrics.Unit `mtx:"% Cache hits, %"`
}

func New(dir string, dc deps.Container) (m *HgtMgr, err error) {
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

    dc.MetricsCollector.Register("Hgt manager", &mtx)

    go m.run(dc)

    return
}

func (m *HgtMgr) GetGrid(rect latlon.Recti, mask []bool) *Grid {
    r := make(chan *Grid)
    m.rq <- request{Recti: rect, replyto: r, mask: mask}
    return <-r
}

func (m *HgtMgr) GetGridAround(ll latlon.LL) deps.HgtGrid {
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

func (g *Grid) PtrMap() []uint64 {
    return g.ptrMap
}

func (g *Grid) EvtReady() uint64 {
    return g.evtReady
}

func (g *Grid) Rect() latlon.Recti {
    return g.rect
}

func (g *Grid) Free() {
    g.mgr.free <- g
}

func (m *HgtMgr) run(dc deps.Container) {
    mc := dc.MetricsCollector
    for {
        select {
            case rq := <-m.rq:
                g := &Grid{
                    ptrMap: make([]uint64, 0, rq.Width * rq.Height),
                    rect: rq.Recti,
                    mgr: m,
                    mask: rq.mask,
                }
                prepq := make([]uint64, 0, rq.Width * rq.Height)
                g.rect.Apply(func (ll latlon.LLi) {
                    if rq.mask[len(g.ptrMap)] == false {
                        g.ptrMap = append(g.ptrMap, 0)
                        return
                    }
                    ll = ll.Wrap()
                    mc.Count(&mtx.CacheRequest)
                    cr, ok := m.cacheMap[ll]
                    if ok {
                        mc.Add(&mtx.CacheHit, 1)
                    } else {
                        mc.Add(&mtx.CacheHit, 0)
                        for le := m.cache.Back(); le != nil; le = le.Prev() {
                            if le.Value.(*cacheRecord).users > 0 {
                                continue
                            }
                            cr = le.Value.(*cacheRecord)
                            ptr := cuhgt.Fetch(ll, m.hgtdir, cr.slot)
                            if ptr == 0 {
                                m.cacheMap[ll] = nil
                                g.mask[len(g.ptrMap)] = false;
                                g.ptrMap = append(g.ptrMap, 0)
                                return
                            } else {
                                delete(m.cacheMap, cr.ll)
                                cr.ll = ll
                                cr.ptr = ptr
                                m.cacheMap[ll] = cr
                                prepq = append(prepq, ptr)
                            }
                            break
                        }
                        if cr == nil {
                            log.Fatal("cache overflow!")
                        }
                    }
                    if cr == nil {
                        g.mask[len(g.ptrMap)] = false;
                        g.ptrMap = append(g.ptrMap, 0)
                        return
                    }
                    cr.users++
                    g.ptrMap = append(g.ptrMap, cr.ptr)
                    m.cache.MoveToFront(cr.le)
                })
                g.evtReady = cuhgt.Prepare(prepq)
                rq.replyto <- g
            case g := <-m.free:
                g.mgr = nil
                i := -1
                g.rect.Apply(func (ll latlon.LLi) {
                    i++
                    if g.mask[i] == false {
                        return
                    }
                    cr, ok := m.cacheMap[ll.Wrap()]
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
