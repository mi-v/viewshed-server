// +build dev

package conf

import "math"

const ERAD = 6371000.0
const MRDLEN = math.Pi * ERAD
const CSLAT = MRDLEN / 180 // cell size along latitude
//const CUTOFF = 150000
const CUTOFF = 200000
const CUTON = 80
const MAXWIDTH = 20000
const TORAD = math.Pi / 180
const HGTDIR = "/projects/vshed/hgt"
const TILEDIR = "/projects/vshed/tiles-dev"
const HGTSLOTS = 100
const LISTEN_PORT = 4003
