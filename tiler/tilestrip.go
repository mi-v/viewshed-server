package tiler

type Corner struct {
    X, Y int
}

type Tile struct {
    Corner
    Z int
    Pix []byte
}

type Rect struct {
    P, Q Corner
}

type StripZoom struct {
    Rect
    Ntiles, Pretiles int
}

type Strip struct {
    buf []byte
    Z []StripZoom
    free func()
    idx, iz int
}

func (r Rect) W() int {
    return r.Q.X - r.P.X
}

func (r Rect) H() int {
    return r.Q.Y - r.P.Y
}

func NewStrip(buf []byte, sz []StripZoom, free func()) *Strip {
    return &Strip{
        buf: buf,
        Z: sz,
        free: free,
    }
}

func (ts *Strip) Free() {
    ts.buf = nil
    ts.free()
    ts.free = nil;
}

func (ts *Strip) Rewind() (Tile, bool) {
    ts.idx = -1
    ts.iz = len(ts.Z)-1
    return ts.Next()
}

const tileBytes = 256 * 256 / 8

func (ts *Strip) Next() (Tile, bool) {
    ts.idx++
    if ts.idx > ts.Z[0].Pretiles {
        return Tile{}, false
    }
    if ts.idx >= ts.Z[ts.iz].Pretiles + ts.Z[ts.iz].Ntiles {
        ts.iz--
    }
    Z := ts.Z[ts.iz]
    ti := ts.idx - Z.Pretiles
    return Tile{
        Corner: Corner{
            Z.P.X + ti % Z.W(),
            Z.P.Y + ti / Z.W(),
        },
        Z: ts.iz,
        Pix: ts.buf[ts.idx * tileBytes : (ts.idx+1) * tileBytes],
    }, true
}
