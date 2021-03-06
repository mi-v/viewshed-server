package tiler

import (
    "sync"
    "log"
    "os"
    "fmt"
    "image"
    "image/color"
    "github.com/mi-v/img1b"
    "github.com/mi-v/img1b/png"
)

type Tiler struct {
    c chan encTask
}

type encTask struct {
    Tile
    dir string
    report *sync.WaitGroup
}

var tasks chan encTask

type encPool struct {
	b *png.EncoderBuffer
}

func (p *encPool) Get() *png.EncoderBuffer {
	return p.b
}

func (p *encPool) Put(b *png.EncoderBuffer) {
	p.b = b
}

func New() (tr *Tiler) {
    tr = &Tiler{make(chan encTask)}
    for i := 0; i < 7; i++ {
        go func() {
            e := png.Encoder{
                BufferPool: &encPool{},
                //CompressionLevel: png.BestSpeed,
            }
            for task := range tr.c {
//task.report.Done()
//continue
                dir := fmt.Sprintf("%s/z%d/%d", task.dir, task.Z, task.X & (1 << task.Z - 1))
                fn := fmt.Sprintf("%s/%d.png", dir, task.Y)
                fd, err := os.Create(fn)
                if err != nil {
                    err = os.MkdirAll(dir, os.ModePerm)
                    fd, err = os.Create(fn)
                    if err != nil {
                        log.Println(err)
                        task.report.Done()
                        continue
                    }
                }
                img := &img1b.Image{
                    Pix: task.Pix,
                    Stride: 256 / 8,
                    Rect: image.Rect(0, 0, 256, 256),
                    Palette: color.Palette{
                        color.Black,
                        color.Transparent,
                    },
                }
                e.Encode(fd, img)
                fd.Close()
                task.report.Done()
            }
        }()
    }
    return tr
}

func (tr *Tiler) Encode(te Tile, dir string, report *sync.WaitGroup) {
    tr.c <- encTask{te, dir, report}
}
