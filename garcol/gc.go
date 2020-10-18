package garcol

import (
    "time"
    "os"
    "fmt"
)

const ttl = 3 * time.Minute

type record struct {
    since time.Time
    timer *time.Timer
}

type request struct {
    name string
    replyto chan bool
}

type GC struct {
    list map[string]record
    kc chan *request
    dc chan string
}

func (gc *GC) preload(root string) error {
    d, err := os.Open(root)
    if err != nil {
        return err
    }
    oldstuff, err := d.Readdirnames(-1)
    d.Close()
    if err != nil {
        return err
    }

    for _, name := range oldstuff {
        if name[0] == '-' || name[0] == '+' {
            gc.Keep(name)
        }
    }

    return nil
}

func (gc *GC) Keep(name string) (there bool) {
    rc := make(chan bool)
    gc.kc <- &request{name, rc}
    return <-rc
}

func New(root string) (*GC, error) {
    gc := &GC{
        list: make(map[string]record),
        kc: make(chan *request),
        dc: make(chan string),
    }

    go func() {
        for {
            select {
            case rq := <-gc.kc:
                rec, ok := gc.list[rq.name]
                if ok {
                    if !rec.timer.Reset(ttl) {
                        <-gc.dc
                    }
                    rq.replyto <- true
                    break
                }
                gc.list[rq.name] = record{
                    since: time.Now(),
                    timer: time.AfterFunc(ttl, func () {
                        gc.dc <- rq.name
                    }),
                }
                rq.replyto <- false

            case del := <-gc.dc:
                deltmp := fmt.Sprintf("del.%d.%s", time.Now().Unix(), del)
                os.Rename(root + "/" + del, root + "/" + deltmp)
                go os.RemoveAll(root + "/" + deltmp)
                delete(gc.list, del)
            }
        }
    }()

    err := gc.preload(root)
    if err != nil {
        return nil, err
    }

    return gc, nil
}
