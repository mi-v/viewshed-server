package img1b

import (
	"image"
	"image/color"
)

type Image struct {
	Pix []uint8
	Stride int
	Rect image.Rectangle
	Palette color.Palette
}

func (p *Image) At(x, y int) color.Color {
	if len(p.Palette) == 0 {
		return nil
	}
	if !(image.Point{x, y}.In(p.Rect)) {
		return p.Palette[0]
	}
	i, b := p.PixBitOffset(x, y)
	return p.Palette[(p.Pix[i] >> b) & 1]
}

func (p *Image) PixBitOffset(x, y int) (ofs, bit int) {
	ofs = (y-p.Rect.Min.Y)*p.Stride + (x-p.Rect.Min.X)/8
	bit = 7 - (x-p.Rect.Min.X) % 8
	return
}

func (p *Image) Bounds() image.Rectangle { return p.Rect }

func (p *Image) ColorModel() color.Model { return p.Palette }

func (p *Image) ColorIndexAt(x, y int) uint8 {
	if !(image.Point{x, y}.In(p.Rect)) {
		return 0
	}
	i, b := p.PixBitOffset(x, y)
	return (p.Pix[i] >> b) & 1
}

func (p *Image) SetColorIndex(x, y int, index uint8) {
	if !(image.Point{x, y}.In(p.Rect)) {
		return
	}
	i, b := p.PixBitOffset(x, y)
	p.Pix[i] |= index << b
}

func New(r image.Rectangle, p color.Palette) *Image {
	w, h := r.Dx(), r.Dy()
	stride := (w+7) / 8
	pix := make([]uint8, stride*h)
	return &Image{pix, stride, r, p}
}
