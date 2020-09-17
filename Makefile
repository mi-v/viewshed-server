NVCC=nvcc
#NVCCFLAGS=-O3 -arch=sm_52 -m64 --ptxas-options=-v --compiler-options '-fPIC' --shared
#NVCCFLAGS=-m64 --ptxas-options=-v --compiler-options '-fPIC' --shared
NVCCFLAGS=-m64 --resource-usage --compiler-options '-fPIC' --shared
VPATH=cuhgt cuvshed cuinit

RDIR = release
PROJ = vshed
RELNUM := $(shell date +%d%H%M)
PDIR = /projects/${PROJ}
DDIR = ${PDIR}/releases/${RELNUM}

build: vshed

#deploy: check-if-master build
deploy: build
	mkdir ${DDIR}
	cp vshed *.so ${DDIR}
	rm ${PDIR}/current
	ln -s ${DDIR} ${PDIR}/current

cuda: libcuhgt.so libcuvshed.so libcuinit.so

lib%.so: %.cu %.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<
	chmod -x $@

check-if-master:
	git branch | fgrep -q '* master'

vshed: $(shell find -type f -name '*.go')
	GOARCH=amd64 go build -ldflags="-s -w" vshed
