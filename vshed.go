package main

import (
    "fmt"
    "log"
    . "vshed/conf"
    "vshed/latlon"
    "vshed/hgtmgr"
    "vshed/cuvshed"
    "time"
    "math"
    "os"
    //"image/png"
    "vshed/img1b/png"
)

func main() {
    hm, err := hgtmgr.New("../hgt/3")
    if err != nil {
        log.Fatal(err)
    }

    var t time.Time

    //ll := latlon.LL{50.339926, 87.748689} // ретр-р
    ll := latlon.LL{49.809202, 86.589432} // Белуха
    r := latlon.RectiFromRadius(ll, CUTOFF / CSLAT + 0.1)
    mask := make([]bool, 0, r.Width * r.Height)
    r.Apply(func (cll latlon.LLi) {
        cutoff := CUTOFF / CSLAT + 0.1
        cllf := cll.Float()
        dY := clamp(ll.Lat, cllf.Lat, cllf.Lat + 1) - ll.Lat
        dX := (clamp(ll.Lon, cllf.Lon, cllf.Lon + 1) - ll.Lon) * math.Cos((ll.LatR() + cllf.LatR()) / 2)
        mask = append(mask, dX * dX + dY * dY < cutoff * cutoff)
    })

    t = time.Now()
    grid := hm.GetGrid(r, mask)
    fmt.Println(grid)
    fmt.Println("GetGrid: ", time.Since(t), "\n")

    t = time.Now()
    ts, _ := cuvshed.TileStrip(ll, 2, grid.Map, grid.Recti)
    fmt.Println("TileStrip: ", time.Since(t), "\n")

    t = time.Now()
    for tl, ok := ts.Rewind(); ok; tl, ok = ts.Next() {
        dir := fmt.Sprintf("tiles/z%d/%d", tl.Z, tl.X)
        fn := fmt.Sprintf("%s/%d.png")
        fd, err := os.Create(fn)
        if err != nil {
            err = os.MkdirAll(dir)
            fd, err := os.Create(fn)
            if err != nil {
                continue
            }
        }
        img := &img1b.Image{
            Pix: tl.Pix,
            Stride: 256 / 8,
            Rect: image.Rect(0, 0, 256, 256),
            Palette: color.Palette{
                color.Black,
                color.White,
            },
        }
        png.Encode(fd, img)
        fd.Close()
    }
    fmt.Println("\nIter: ", time.Since(t), "\n")

    return

    /*t = time.Now()
    img := cuvshed.Image(ll, 2, grid.Map, grid.Recti)
    fmt.Println("Image: ", time.Since(t), "\n")

    t = time.Now()
    f, _ := os.Create("Img.png")
    png.Encode(f, img)
    fmt.Println("encode: ", time.Since(t), "\n")*/
}

func clamp(v, min, max float64) float64 {
    return math.Min(math.Max(v, min), max)
}
