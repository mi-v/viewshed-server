package tiler

//import "log"

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

var _tileRect = Rect{
    Corner{0, 0},
    Corner{256, 256},
}
const tileBytes = 256 * 256 / 8

type StripZoom struct {
    Rect
    Ntiles, Pretiles int
}

type Strip struct {
    buf []byte
    Z []StripZoom
    idx, iz int
}

func (r Rect) W() int {
    return r.Q.X - r.P.X
}

func (r Rect) H() int {
    return r.Q.Y - r.P.Y
}

func (r Rect) WH() int {
    return r.W() * r.H()
}

func (r Rect) Index(i int) Corner {
    return Corner{
        r.P.X + i % r.W(),
        r.P.Y + i / r.W(),
    }
}

func (r Rect) IndexOf(p Corner) int {
    return (p.Y - r.P.Y) * r.W() + (p.X - r.P.X)
}

func (r Rect) WrapX(z int) Rect {
    return Rect{
        P: Corner{r.P.X & (1 << z - 1), r.P.Y},
        Q: Corner{r.Q.X & (1 << z - 1), r.Q.Y},
    }
}

func NewStrip(buf []byte, sz []StripZoom) *Strip {
    return &Strip{
        buf: buf,
        Z: sz,
    }
}

func (ts *Strip) MaxZ() int {
    return len(ts.Z) - 1;
}

func (ts *Strip) Rewind() (Tile, bool) {
    ts.idx = -1
    ts.iz = len(ts.Z)-1
    return ts.Next()
}

func (ts *Strip) QueryPx(pos Corner, z int) bool {
    Z := ts.Z[z]
    tlPos := Corner{pos.X >> 8, pos.Y >> 8}
    pxPos := Corner{pos.X & 255, pos.Y & 255}
    tlIdx := Z.IndexOf(tlPos)
    if z == 0 {
        tlIdx = 0
    }
    pxIdx := _tileRect.IndexOf(pxPos)
    return (ts.buf[(Z.Pretiles + tlIdx) * tileBytes + pxIdx >> 3] & (0x80 >> (pxIdx & 7))) != 0
}

func (ts *Strip) Next() (Tile, bool) {
    var pos Corner
    for {
        ts.idx++
        if ts.idx > ts.Z[0].Pretiles {
            return Tile{}, false
        }
        if ts.idx >= ts.Z[ts.iz].Pretiles + ts.Z[ts.iz].Ntiles {
            ts.iz--
        }

        Z := ts.Z[ts.iz]
        ti := ts.idx - Z.Pretiles
        pos = Z.Index(ti)

        // a pixel 8 levels up is the current tile
        if ts.iz < 8 || ts.QueryPx(pos, ts.iz-8) {
            break
        }
    }
    return Tile{
        Corner: pos,
        Z: ts.iz,
        Pix: ts.buf[ts.idx * tileBytes : (ts.idx+1) * tileBytes],
    }, true
}
