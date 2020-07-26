NVCC=nvcc
#NVCCFLAGS=-O3 -arch=sm_52 -m64 --ptxas-options=-v --compiler-options '-fPIC' --shared
#NVCCFLAGS=-m64 --ptxas-options=-v --compiler-options '-fPIC' --shared
NVCCFLAGS=-m64 --resource-usage --compiler-options '-fPIC' --shared
VPATH=cuhgt cuvshed

cuda: libcuhgt.so libcuvshed.so

lib%.so: %.cu %.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<
	chmod -x $@
