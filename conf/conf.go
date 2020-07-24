package conf

import "math"

const ERAD = 6371000.0
const MRDLEN = math.Pi * ERAD
const CSLAT = MRDLEN / 180 // cell size along latitude
//const CUTOFF = 150000
const CUTOFF = 200000
const CUTON = 80
const MAXZOOM = 10
const TORAD = math.Pi / 180
