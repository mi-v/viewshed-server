package main

import (
    "os"
    "fmt"
    "log"
    . "vshed/conf"
    "vshed/latlon"
    "vshed/hgtmgr"
    "vshed/cuvshed"
    "vshed/tiler"
    "encoding/json"
    "time"
    //"math"
    "sync"
    //"runtime/pprof"
    "net/http"
    _ "net/http/pprof"
    "regexp"
    "strconv"
    "io/ioutil"
)

func main() {
    /*pf, _ := os.Create("vs.prof");
    pprof.StartCPUProfile(pf)
    defer pprof.StopCPUProfile()*/

    hm, err := hgtmgr.New(HGTDIR)
    if err != nil {
        log.Fatal(err)
    }

    originHostRE := regexp.MustCompile(`(votetovid\.ru|mapofviews\.com)(:|/|$)`)

    http.HandleFunc("/", func (w http.ResponseWriter, r *http.Request) {
        if r.Method != "GET" {
            http.Error(w, "Unsupported method", http.StatusMethodNotAllowed)
            return
        }

        w.Header().Set("Content-Type", "application/json; charset=utf-8")

        if originHostRE.MatchString(r.Header.Get("Origin")) {
            w.Header().Set("Access-Control-Allow-Origin", r.Header.Get("Origin"))
        }

        qlat := r.URL.Query().Get("lat")
        if qlat[0] == ' ' {
            qlat = qlat[1:]
        }
        qlon := r.URL.Query().Get("lon")
        if qlon[0] == ' ' {
            qlon = qlon[1:]
        }
        lat, errlat := strconv.ParseFloat(qlat, 64)
        lon, errlon := strconv.ParseFloat(qlon, 64)
        if errlat != nil || errlon != nil || lat > 85 || lat < -85 {
            http.Error(w, "Invalid parameters", 400)
            return
        }
        ll := latlon.LL{lat, lon}.Wrap()

        obsAh64, errAh := strconv.ParseInt(r.URL.Query().Get("ah"), 10, 32)
        obsBh64, errBh := strconv.ParseInt(r.URL.Query().Get("bh"), 10, 32)
        obsAh, obsBh := int(obsAh64), int(obsBh64)
        if errAh != nil || errBh != nil {
            http.Error(w, "Invalid parameters", 400)
            return
        }

        tilepath := fmt.Sprintf(
            "%+08.4f,%+09.4f,%dah,%dbh",
            lat,
            lon,
            obsAh,
            obsBh,
        )
        tiledir := TILEDIR + "/" + tilepath

        jf, err := os.Open(tiledir + "/layout.json")
        if err == nil {
            defer jf.Close()
            now := time.Now()
            os.Chtimes(jf.Name(), now, now)
            http.ServeContent(w, r, "", time.Time{}, jf)
            return
        }

        response := struct {
            Tilemask [][]byte `json:"tmap,omitempty"`
            Zrects []tiler.Rect `json:"zrct,omitempty"`
            Tilepath string `json:"tpth,omitempty"`
            Zlim int `json:"zlim,omitempty"`
        }{}

        defer func() {
            j, _ := json.Marshal(response);
            ioutil.WriteFile(tiledir + "/layout.json", j, 0666)
            w.Write(j)
        }()

        var t time.Time

        t = time.Now()
        grid := hm.GetGridAround(ll)
        fmt.Println("GetGrid: ", time.Since(t))

        t = time.Now()
        ts, err := cuvshed.TileStrip(ll, obsAh, obsBh, grid.Map, grid.Recti)
        grid.Free()
        if err != nil {
            log.Println(err)
            http.Error(w, "Server error", 500)
            return
        }
        fmt.Println("TileStrip: ", time.Since(t))

        t = time.Now()
        tilemask := make([][]byte, ts.MaxZ()-7)
        for z, Z := range ts.Z {
            if z > 7 {
                tilemask[z-8] = make([]byte, (Z.WH()+7)/8)
            }
            response.Zrects = append(response.Zrects, Z.Rect)
        }
        var wg sync.WaitGroup
        for tl, ok := ts.Rewind(); ok; tl, ok = ts.Next() {
            wg.Add(1)
            tl.Encode(tiledir, &wg)
            if tl.Z > 7 {
                tlIdx := ts.Z[tl.Z].IndexOf(tl.Corner)
                tilemask[tl.Z-8][tlIdx/8] |= 1 << (tlIdx%8)
            }
        }
        wg.Wait()
        ts.Free()
        fmt.Println("Tile cutting: ", time.Since(t))
        response.Tilemask = tilemask

        response.Tilepath = tilepath
        response.Zlim = ts.MaxZ();

        return

        /*t = time.Now()
        img := cuvshed.Image(ll, 2, grid.Map, grid.Recti)
        fmt.Println("Image: ", time.Since(t), "\n")

        t = time.Now()
        f, _ := os.Create("Img.png")
        png.Encode(f, img)
        fmt.Println("encode: ", time.Since(t), "\n")*/
    })

    log.Fatal(http.ListenAndServe(":3003", nil))
}
