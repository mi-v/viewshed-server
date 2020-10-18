package main

import (
    "os"
    "fmt"
    "log"
    . "vshed/conf"
    "vshed/deps"
    "vshed/metrics"
    "vshed/latlon"
    "vshed/hgtmgr"
    "vshed/tiler"
    "vshed/cuvshed"
    "vshed/garcol"
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
    "flag"
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

var mtx struct {
    HttpRequest metrics.Unit `mtx:". HTTP requests"`
    HttpFileResponse metrics.Unit `mtx:". HTTP responses from a file"`
    TaskCanceled metrics.Unit `mtx:". Canceled tasks"`
    TaskNew metrics.Unit `mtx:". New tasks queued"`
    TaskQLen metrics.Unit `mtx:"$ Task queue length avg"`
    TaskQRTO metrics.Unit `mtx:". Task quick reply timeouts"`
    TaskDone metrics.Unit `mtx:". Tasks finished"`
    TimeGetGrid metrics.Unit `mtx:"$ Grid assembly time, ms"`
    TimeCompute metrics.Unit `mtx:"$ Computation time, ms"`
    TimeCutTiles metrics.Unit `mtx:"$ Tile encoding time, ms"`
    TilesWritten metrics.Unit `mtx:"$ Tiles written avg/rq"`
    TilesSkipped metrics.Unit `mtx:"$ Tiles skipped avg/rq"`
}

func main() {
    /*pf, _ := os.Create("vs.prof");
    pprof.StartCPUProfile(pf)
    defer pprof.StopCPUProfile()*/

    var port int
    flag.IntVar(&port, "port", LISTEN_PORT, "port number")

    var err error
    dc := deps.Container{}

    dc.MetricsCollector = metrics.NewCollector()
    dc.MetricsCollector.Register("Main module", &mtx)

    dc.Tiler = tiler.New()

    dc.HgtMgr, err = hgtmgr.New(HGTDIR, dc)
    if err != nil {
        log.Fatal(err)
    }

    mc := dc.MetricsCollector

    originHostRE := regexp.MustCompile(`(votetovid\.ru|mapofviews\.com|sauropod\.xyz)(:|/|$)`)

    tasks := make(chan *task)
    go qmgr(tasks, dc)

    gc, err := garcol.New(TILEDIR)
    if err != nil {
        log.Fatal(err)
    }

    http.HandleFunc("/favicon.ico", func (w http.ResponseWriter, r *http.Request) {})

    http.HandleFunc("/", func (w http.ResponseWriter, r *http.Request) {
        mc.Add(&mtx.HttpRequest, 0)
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

        if _, keep := r.URL.Query()["keep"]; keep {
            gc.Keep(tilepath)
            return
        }

        found := gc.Keep(tilepath)
        if found {
            jf, err := os.Open(TILEDIR + "/" + tilepath + "/layout.json")
            if err == nil {
                mc.Add(&mtx.HttpFileResponse, 0)
                defer jf.Close()
                now := time.Now()
                os.Chtimes(jf.Name(), now, now)
                http.ServeContent(w, r, "", time.Time{}, jf)
                return
            }
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

    http.Handle("/mtx", dc.MetricsCollector)

    log.Println("Started at", port)
    log.Fatal(http.ListenAndServe(":" + strconv.Itoa(port), nil))
}

func worker(tasks chan *task, dc deps.Container) {
    ctx := cuvshed.MakeCtx()
    mc := dc.MetricsCollector

    for tk := range tasks {
        r := response{taskId: tk.taskId}

        var t time.Time

        t = time.Now()
        grid := dc.HgtMgr.GetGridAround(tk.ll)
        mc.Add(&mtx.TimeGetGrid, int(time.Since(t).Milliseconds()))

        t = time.Now()
        ts, err := cuvshed.TileStrip(ctx, tk.ll, tk.obsAh, tk.obsBh, grid)
        grid.Free()
        if err != nil {
            log.Println(err)
            r.err = true
            tk.replyto <- &r
            continue
        }
        mc.Add(&mtx.TimeCompute, int(time.Since(t).Milliseconds()))

        t = time.Now()
        cnt := 0
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
            cnt++
            dc.Tiler.Encode(tl, tiledir, &wg)
            if tl.Z > 7 {
                tlIdx := ts.Z[tl.Z].IndexOf(tl.Corner)
                tilemask[tl.Z-8][tlIdx/8] |= 1 << (tlIdx%8)
            }
        }
        wg.Wait()
        mc.Add(&mtx.TimeCutTiles, int(time.Since(t).Milliseconds()))
        mc.Add(&mtx.TilesWritten, cnt)
        mc.Add(&mtx.TilesSkipped, ts.Z[0].Pretiles + 1 - cnt)

        r.Tilemask = tilemask
        r.Tilepath = tk.tilepath
        r.Zlim = ts.MaxZ();

        r.json, _ = json.Marshal(r);
        ioutil.WriteFile(tiledir + "/layout.json", r.json, 0666)

        tk.replyto <- &r
    }
}

func qmgr(tkin chan *task, dc deps.Container) {
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
    mc := dc.MetricsCollector

    for i := 0; i < 2; i++ {
        go worker(tkout, dc)
    }

    for {
        tkout := tkout // a local shadow
        var tk *task

        for len(q) > 0 && q[0].ts.Add(2*time.Second).Before(time.Now()) {
            delete(qmap, q[0].taskId) // shake off canceled tasks
            delete(rsout, q[0].taskId)
            q = q[1:]
            dqd++
            mc.Add(&mtx.TaskCanceled, 0)
        }

        if len(q) > 0 { // if we have something queued
            tk = q[0].task // get it ready
        } else {
            tkout = nil // otherwise we'll be poking nil channel instead
        }

        select {
        case tk := <-tkin: // got a new task
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
                mc.Add(&mtx.TaskNew, 0)
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
                mc.Add(&mtx.TaskQLen, len(q))
            }

        case qe := <-qrtc: // a quick reply timed out
            mc.Add(&mtx.TaskQRTO, 0)
            pos := qe.pos - dqd
            if pos < 1 {
                pos = 1
            }
            select {
                case rsout[qe.taskId] <- &response{Qp: pos}:
                default:
            }

        case tkout <- tk: // started a task, dequeue it
            q = q[1:]
            dqd++

        case rs := <-rsin: // got a worker response
            if qmap[rs.taskId] != nil {
                qmap[rs.taskId].qrt.Stop()
            }
            select { // try to forward it
                case rsout[rs.taskId] <- rs:
                default:
            }
            delete(qmap, rs.taskId)
            delete(rsout, rs.taskId)
            mc.Add(&mtx.TaskDone, 0)
        }
    }
}
