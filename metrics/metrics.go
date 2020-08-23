package metrics

import (
    "time"
    "reflect"
    "fmt"
    "strings"
    "net/http"
)

const depth = 8
const numPeriods = 8
var periods = [numPeriods]int {1, 5, 15, 30, 60, 4*60, 8*60, 24*60}

const (
    fmtSum = iota
    fmtCount // .
    fmtAvg // $
    fmtPercent // %
)

type Unit struct {
    title string
    data [numPeriods][depth]int
    count [numPeriods][depth]int
    fmt int
}

type sample struct {
    *Unit
    value int
}

type sheet struct {
    title string
    data []*Unit
}

type Collector struct {
    c chan sample
    book []sheet
    ul []*Unit
}

func (u *Unit) shift(pi int) {
    for i := depth-1; i > 0; i-- {
        u.data[pi][i] = u.data[pi][i-1]
        u.count[pi][i] = u.count[pi][i-1]
    }
    u.data[pi][0] = 0
    u.count[pi][0] = 0
}

func (u *Unit) value(pi, d int) float64 {
    f := float64(u.data[pi][d])
    switch u.fmt {
        case fmtCount: f = float64(u.count[pi][d])
        case fmtAvg: f /= float64(u.count[pi][d])
        case fmtPercent: f *= 100 / float64(u.count[pi][d])
    }
    return f
}

func NewCollector() (cr *Collector) {
    cr = &Collector{c: make(chan sample)}

    tr := time.NewTicker(time.Hour)
    go func() {
        time.Sleep(time.Now().Truncate(time.Minute).Add(time.Minute).Sub(time.Now()))
        tr.Reset(time.Minute)
    }()

    go func() {
        for {
            select {
            case s := <-cr.c:
                for i := 0; i < numPeriods; i++ {
                    s.data[i][0] += s.value
                    s.count[i][0] ++
                }
            case t := <- tr.C:
                for _, up := range cr.ul {
                    for pi, p := range periods {
                        if (int(t.Unix()) / 60) % p == 0 {
                            up.shift(pi)
                        }
                    }
                }
            }
        }
    }()
    return cr
}

func (cr *Collector) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintln(w, "<!DOCTYPE html><table>")

    for _, st := range cr.book {
        fmt.Fprintf(w, "<tr><th colspan='%d'>%s</th></tr>", numPeriods + 1, st.title)

        fmt.Fprint(w, "<tr><th></th>")
        for _, p := range periods {
            ds := (time.Duration(p) * time.Minute).String()
            for i, c := range ds {
                if i > 0 && c == '0' && ds[i-1] > '9' {
                    ds = ds[:i]
                    break
                }
            }
            fmt.Fprintf(w, "<th>%s</th>", ds)
        }
        fmt.Fprint(w, "</tr>")

        for _, u := range st.data {
            fmt.Fprintf(w, "<tr><th>%s</th>", u.title)
            for pi := range periods {
                log := ""
                for d := 1; d < depth; d++ {
                    log += fmt.Sprintf("&#010;%.0f", u.value(pi, d))
                }
                log = log[6:]

                fmt.Fprintf(w, "<td title='%s'>%.0f</td>", log, u.value(pi, 0))
            }
            fmt.Fprint(w, "</tr>\n");
        }
    }

    fmt.Fprintln(w, "</table>\n")
}

func (cr *Collector) Add(u *Unit, v int) {
    cr.c <- sample{u, v}
}

func (cr *Collector) Count(u *Unit) {
    cr.Add(u, 0)
}

func (cr *Collector) Register(title string, units interface{}) {
    var st *sheet
    for i := range cr.book {
        if cr.book[i].title == title {
            st = &cr.book[i]
            break
        }
    }
    if st == nil { // not found in the book
        cr.book = append(cr.book, sheet{title, nil})
        st = &cr.book[len(cr.book)-1]
    }

    usr := reflect.ValueOf(units).Elem()
    for i := 0; i < usr.NumField(); i++ {
        u := usr.Field(i).Addr().Interface().(*Unit)
        if i == 0 && len(st.data) > 0 && st.data[0] == u {
            return // this sheet is already registered
        }
        u.title = usr.Type().Field(i).Tag.Get("mtx")
        if len(u.title) > 0 {
            switch u.title[0] {
            case '.':
                u.fmt = fmtCount
            case '$':
                u.fmt = fmtAvg
            case '%':
                u.fmt = fmtPercent
            }
            if u.fmt != fmtSum {
                u.title = strings.TrimSpace(u.title[1:])
            }
        }

        cr.ul = append(cr.ul, u)
        st.data = append(st.data, u)
    }
}
