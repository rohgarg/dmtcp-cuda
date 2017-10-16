# To demonstrate, do:  make check    [Checkpoints every 5 seconds]

# The name will be the same as the current directory name.
NAME=${shell basename $$PWD}

# By default, your resulting library will have this name.
LIBNAME=libdmtcp_${NAME}

# As you add new files to your hijack library, add the object file names here.
LIBOBJS = cuda-plugin.o cuda-wrappers.o cuda-common-utils.o cuda-uvm-utils.o cuda-uvm-wrappers.o

CC=nvcc
CXX=nvcc

# Modify if your DMTCP_ROOT is located elsewhere.
ifndef DMTCP_ROOT
  DMTCP_ROOT=../..
endif
DMTCP_INCLUDE=${DMTCP_ROOT}/include
JALIB_INCLUDE=${DMTCP_ROOT}/jalib/

CFLAGS += -I${DMTCP_INCLUDE} -I${JALIB_INCLUDE} -DDMTCP -ccbin /usr/bin/gcc-5 -Xcompiler -fPIC -c -g -O0
CXXFLAGS += -I${DMTCP_INCLUDE} -I${JALIB_INCLUDE} -DDMTCP -ccbin /usr/bin/g++-5 -Xcompiler -fPIC -c -g -O0 -cudart shared

# enable this for debugging the plugin
# CFLAGS += -DCUDA_PLUGIN_DEBUG
# CXXFLAGS += -DCUDA_PLUGIN_DEBUG

DEMO_PORT=7781

default: ${LIBNAME}.so

vi:
	vim cuda-plugin.cpp

cudaproxy: cudaproxy.cu
	nvcc -cudart shared -DSTANDALONE -I. -g -O0 -std c++11 -ccbin /usr/bin/g++-5 -o $@ $<

test1: test/test1.cu
	cd test; nvcc -cudart shared -g -O0 -std c++11 -ccbin /usr/bin/g++-5 -o $@ $@.cu

test2: test/test2.cu
	cd test; nvcc -cudart shared -g -O0 -std c++11 -ccbin /usr/bin/g++-5 -o $@ $@.cu

test2.1: test/test2.1.cu
	cd test; nvcc -cudart shared -g -O0 -std c++11 -ccbin /usr/bin/g++-5 -o $@ $@.cu

test7: test/test7.cu
		cd test; nvcc -arch sm_30 -cudart shared -g -O0 -std c++11 -ccbin /usr/bin/g++-5 -o $@ $@.cu

test7.1: test/test7.1.cu
		cd test; nvcc -arch sm_30 -cudart shared -g -O0 -std c++11 -ccbin /usr/bin/g++-5 -o $@ $@.cu

run2.1: ${LIBNAME}.so cudaproxy test2.1
	../../bin/dmtcp_launch -j  --with-plugin ./libdmtcp_cuda.so test/test2.1

run7: ${LIBNAME}.so cudaproxy test7
	../../bin/dmtcp_launch -j  --with-plugin ./libdmtcp_cuda.so test/test7

run7.1: ${LIBNAME}.so cudaproxy test7.1
	../../bin/dmtcp_launch -j  --with-plugin ./libdmtcp_cuda.so test/test7.1

restart:
	../../bin/dmtcp_restart -j ckpt*.dmtcp

check: ${LIBNAME}.so cudaproxy test/test1
	# Note that full path of plugin (using $$PWD in this case) is required.
	${DMTCP_ROOT}/bin/dmtcp_launch --port ${DEMO_PORT} -i 5 \
	  --with-plugin $$PWD/${LIBNAME}.so test/test1

# We link the library using C++ for compatibility with the main libdmtcp.so
${LIBNAME}.so: ${LIBOBJS}
	${CXX} -cudart shared -shared -o $@ $^

.c.o:
	${CC} ${CFLAGS} -o $@ $<
.cpp.o:
	${CXX} ${CXXFLAGS} -o $@ $<

tidy:
	rm -f *~ .*.swp dmtcp_restart_script*.sh ckpt_*.dmtcp cudaSysCallsLog

clean:
	rm -f ${LIBOBJS} ${LIBNAME}.so test2 test/test2.1 cudaproxy cudaSysCallsLog

distclean: clean
	rm -f ${LIBNAME}.so *~ .*.swp dmtcp_restart_script*.sh ckpt_*.dmtcp

dist: distclean
	dir=`basename $$PWD`; cd ..; \
	  tar czvf $$dir.tar.gz --exclude-vcs ./$$dir
	dir=`basename $$PWD`; ls -l ../$$dir.tar.gz

.PHONY: default clean dist distclean
