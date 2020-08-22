package metrics

import (
    "time"
    "reflect"
    "fmt"
)

const depth = 8
const numPeriods = 8
var periods = [numPeriods]int {1, 5, 15, 30, 60, 4*60, 8*60, 24*60}

type Unit struct {
    title string
    data [numPeriods][depth]int
    count [numPeriods][depth]int
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

func NewCollector() (cr *Collector) {
    cr = &Collector{c: make(chan sample)}
    var tc <-chan time.Time

    go func() {
        //time.Sleep(time.Now().Truncate(time.Minute).Add(time.Minute).Sub(time.Now()))
        //tc = time.Tick(time.Minute)
        time.Sleep(time.Now().Truncate(time.Second).Add(time.Second).Sub(time.Now()))
        tc = time.Tick(time.Second)
    }()
    fmt.Println(tc)

    go func() {
        for {
            select {
            case s := <-cr.c:
                for i := 0; i < numPeriods; i++ {
                    s.data[i][0] += s.value
                    s.count[i][0] ++
                }
            case t := <- tc:
                //fmt.Println(t)
                for _, up := range cr.ul {
                    for pi, p := range periods {
                        //if (int(t.Unix()) / 60) % p == 0 {
                        if (int(t.Unix())) % p == 0 {
                            up.shift(pi)
                        }
                    }
                }

                for _, st := range cr.book {
                    fmt.Println(st.title)
                    for _, u := range st.data {
                        fmt.Printf("  %s", u.title)
                        for pi, p := range periods {
                            fmt.Printf("  %d:", p)
                            //for d := 0; d < depth; d++ {
                            for d := 0; d < 1; d++ {
                                if u.data[pi][d] == 0 {
                                    fmt.Printf(" %d", u.count[pi][d])
                                    continue
                                }
                                fmt.Printf(" %d/%d=%d", u.data[pi][d], u.count[pi][d], u.data[pi][d] / u.count[pi][d])
                            }
                        }
                        fmt.Print("\n");
                    }
                }
            }
        }
    }()
    return cr
}

func (cr *Collector) Add(u *Unit, v int) {
    cr.c <- sample{u, v}
    //cr.c <- sample{Unit: u, value: v}
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
        u.title = usr.Type().Field(i).Tag.Get("mtx")
        cr.ul = append(cr.ul, u)
        st.data = append(st.data, u)
    }
}
