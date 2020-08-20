package main

import (
    "os"
    "fmt"
    "log"
    . "vshed/conf"
    "vshed/latlon"
    "vshed/hgtmgr"
    "vshed/tiler"
    "vshed/cuvshed"
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

type taskId struct {
    ll latlon.LL
    obsAh, obsBh int
}

type response struct {
    Tilemask [][]byte `json:"tmap,omitempty"`
    Zrects []tiler.Rect `json:"zrct,omitempty"`
    Tilepath string `json:"tpth,omitempty"`
    Zlim int `json:"zlim,omitempty"`
    Qp int `json:"qp,omitempty"`
    err bool
    taskId
    json []byte
}

type task struct {
    taskId
    tilepath string
    replyto chan *response
}

var hm *hgtmgr.HgtMgr

func main() {
    /*pf, _ := os.Create("vs.prof");
    pprof.StartCPUProfile(pf)
    defer pprof.StopCPUProfile()*/

    var err error

    hm, err = hgtmgr.New(HGTDIR)
    if err != nil {
        log.Fatal(err)
    }

    originHostRE := regexp.MustCompile(`(votetovid\.ru|mapofviews\.com)(:|/|$)`)

    tasks := make(chan *task)
    go qmgr(tasks)

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

        rc := make(chan *response)
        tasks <- &task{
            taskId: taskId{
                ll: latlon.LL{lat, lon}.Wrap(),
                obsAh: obsAh, obsBh: obsBh,
            },
            tilepath: tilepath,
            replyto: rc,
        }

        rs := <-rc
        if rs.err {
            http.Error(w, "Server error", 500)
            return
        }

        if rs.json == nil {
            rs.json, _ = json.Marshal(rs);
        }

        w.Write(rs.json)

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

func worker(tasks chan *task) {
    ctx := cuvshed.MakeCtx()

    for tk := range tasks {
        r := response{taskId: tk.taskId}

        var t time.Time

        t = time.Now()
        grid := hm.GetGridAround(tk.ll)
        fmt.Println("GetGrid: ", time.Since(t))
        //fmt.Printf("ll: %+v  g: %+v\n", tk.ll, grid)

        t = time.Now()
        ts, err := cuvshed.TileStrip(ctx, tk.ll, tk.obsAh, tk.obsBh, grid.Map, grid.Recti, grid.EvtReady)
        grid.Free()
        if err != nil {
            log.Println(err)
            r.err = true
            tk.replyto <- &r
            continue
        }
        fmt.Println("TileStrip: ", time.Since(t))

        t = time.Now()
        tilemask := make([][]byte, ts.MaxZ()-7)
        for z, Z := range ts.Z {
            if z > 7 {
                tilemask[z-8] = make([]byte, (Z.WH()+7)/8)
            }
            r.Zrects = append(r.Zrects, Z.Rect.WrapX(z))
        }

        var wg sync.WaitGroup
        tiledir := TILEDIR + "/" + tk.tilepath
        for tl, ok := ts.Rewind(); ok; tl, ok = ts.Next() {
            wg.Add(1)
            tl.Encode(tiledir, &wg)
            if tl.Z > 7 {
                tlIdx := ts.Z[tl.Z].IndexOf(tl.Corner)
                tilemask[tl.Z-8][tlIdx/8] |= 1 << (tlIdx%8)
            }
        }
        wg.Wait()
        fmt.Println("Tile cutting: ", time.Since(t))
        r.Tilemask = tilemask

        r.Tilepath = tk.tilepath
        r.Zlim = ts.MaxZ();

        r.json, _ = json.Marshal(r);
        ioutil.WriteFile(tiledir + "/layout.json", r.json, 0666)

        tk.replyto <- &r
    }
}

func qmgr(tkin chan *task) {
    type qentry struct {
        *task
        pos int
        ts time.Time
        qrt *time.Timer
    }

    var q []*qentry
    tkout := make(chan *task)
    rsin := make(chan *response)
    qmap := make(map[taskId]*qentry)
    rsout := make(map[taskId]chan *response)
    dqd := 0 // dequeued tasks, supposed to be one less than head pos
    qrtc := make(chan *qentry) // quick reply timeouts channel

    for i := 0; i < 2; i++ {
        go worker(tkout)
    }

    for {
        tkout := tkout // a local shadow
        var tk *task

        for len(q) > 0 && q[0].ts.Add(2*time.Second).Before(time.Now()) {
            fmt.Printf("canceled: %+v\n", q[0].taskId);
            delete(qmap, q[0].taskId) // shake off canceled tasks
            delete(rsout, q[0].taskId)
            q = q[1:]
            dqd++
        }

        if len(q) > 0 { // if we have something queued
            tk = q[0].task // get it ready
        } else {
            tkout = nil // otherwise we'll be poking nil channel instead
        }

        select {
        case tk := <-tkin: // got a new task
            fmt.Printf("got a task: %+v\n", *tk);
            qe, ok := qmap[tk.taskId] // is it already queued?
            if ok {
                qe.ts = time.Now() // then update its timestamp,
                pos := qe.pos - dqd // , get its position
                if pos < 1 {
                    pos = 1
                }
                select {
                    case tk.replyto <- &response{Qp: pos}: // and reply
                    default: // but no worries if we can't
                }

            } else { // if it's not then queue it
                rsout[tk.taskId] = tk.replyto // save the task's response channel
                tk.replyto = rsin // and substitute our own
                var qe *qentry
                qe = &qentry{
                    task: tk,
                    pos: dqd + len(q) + 1,
                    ts: time.Now(),
                    qrt: time.AfterFunc( // start a timer to notify of a quick reply timeout
                        500 * time.Millisecond,
                        func () {qrtc <- qe},
                    ),
                }
                qmap[qe.taskId] = qe
                q = append(q, qe)
            }

        case qe := <-qrtc: // a quick reply timed out
            fmt.Printf("quick reply timed out: %+v\n", *qe);
            pos := qe.pos - dqd
            if pos < 1 {
                pos = 1
            }
            select {
                case rsout[qe.taskId] <- &response{Qp: pos}:
                default:
            }

        case tkout <- tk: // started a task, dequeue it
            fmt.Printf("task started: %+v\n", *tk);
            q = q[1:]
            dqd++

        case rs := <-rsin: // got a worker response
            fmt.Printf("worker response on %+v\n", rs.taskId);
            if qmap[rs.taskId] != nil {
                qmap[rs.taskId].qrt.Stop()
            }
            select { // try to forward it
                case rsout[rs.taskId] <- rs:
                default:
            }
            delete(qmap, rs.taskId)
            delete(rsout, rs.taskId)
        }
        fmt.Printf("q:%+v  qmap:%+v  rsout:%+v  dqd:%+v\n", q, qmap, rsout, dqd);
    }
}
