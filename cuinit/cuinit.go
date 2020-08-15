package cuinit

// #include "cuinit.h"
// void cuinitInit();
// #cgo LDFLAGS: -L../ -lcuinit
import "C"
import "log"

func init() {
    log.Println("cuinit init")
    C.cuinitInit()
}
