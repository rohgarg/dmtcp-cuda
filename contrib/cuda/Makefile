# To demonstrate, do:  make check    [Checkpoints every 5 seconds]

# The name will be the same as the current directory name.
NAME=${shell basename $$PWD}

# By default, your resulting library will have this name.
LIBNAME=libdmtcp_${NAME}.so

# As you add new files to your hijack library, add the object file names here.
LIBOBJS = cuda-plugin.o cuda-wrappers.o cuda-common-utils.o cuda-uvm-utils.o cuda-uvm-wrappers.o

AUTO_GENERATE_FILES = python-auto-generate/cudawrappers.icpp \
	python-auto-generate/cudaproxy.icpp

CC=nvcc
CXX=nvcc

# *************** Make sure 'gcc --version' is appropriate
# READ FILE RESULTING FROM:
#   PREFIX=`which nvcc` && echo `dirname $PREFIX`/../include/host_config.h
#      (or similar file) and search for '#if defined(__GNUC__)'
#      to see what GNU versions are supported by that CUDA.
# Then check 'g++ --version' for compatibility
NVCC_CC=gcc
NVCC_CXX=g++
NVCC_ROOT=$(abspath $(dir $(abspath $(dir $(shell which nvcc)))))

# Modify if your DMTCP_ROOT is located elsewhere.
ifndef DMTCP_ROOT
  DMTCP_ROOT=../..
endif
DMTCP_INCLUDE=${DMTCP_ROOT}/include
JALIB_INCLUDE=${DMTCP_ROOT}/jalib/


# **********************************************************
# Avoid warning: nvcc warning : The 'compute_10', and 'sm_10' architectures are deprecated, and may be removed in a future release
CXXFLAGS += -arch sm_30
# **********************************************************

# CFLAGS += -I${DMTCP_INCLUDE} -I${JALIB_INCLUDE} -I${DMTCP_ROOT}/src -DDMTCP -ccbin ${NVCC_CC} -Xcompiler -fPIC -c -g -O0
# CXXFLAGS += -I${DMTCP_INCLUDE} -I${JALIB_INCLUDE} -I${DMTCP_ROOT}/src -DDMTCP -ccbin /usr/bin/g++-5 -Xcompiler -fPIC -c -g -O0 -cudart shared
CFLAGS += -I${DMTCP_INCLUDE} -I${JALIB_INCLUDE} -I${DMTCP_ROOT}/src -DDMTCP -ccbin ${NVCC_CC} -Xcompiler -fPIC -c -g -O0
CXXFLAGS += -I${DMTCP_INCLUDE} -I${JALIB_INCLUDE} -I${DMTCP_ROOT}/src -DDMTCP -ccbin ${NVCC_CXX} -Xcompiler -fPIC -c -g -O0 -cudart shared

# enable this for debugging the plugin
# CFLAGS += -DCUDA_PLUGIN_DEBUG
# CXXFLAGS += -DCUDA_PLUGIN_DEBUG

DEMO_PORT=7781

default: ${LIBNAME} libcudaproxy.so

# cudaproxy.icu cudawrappers.icpp cuda_plugin.h created in subdirectory
# A symbolic link to there from main directory allows those files to be included
${AUTO_GENERATE_FILES}: python-auto-generate/cuda-wrappers.py
	cd python-auto-generate && make

vi:
	vim cuda-plugin.cpp

libcudaproxy.so: cudaproxy.cu python-auto-generate/cudaproxy.icu
	nvcc -cudart shared -DSTANDALONE -I. -I${DMTCP_INCLUDE} -I${JALIB_INCLUDE} -I${DMTCP_ROOT}/src -Xcompiler -fPIC -g -O0 -shared -o $@ ../../src/libjalib.a -ldl $<

# tests ---------------------------------------

# The executable (e.g., test/test1) runs as the primary application process.
#   It forks a child process that re-execs itself, but with
#   the environment variable LD_PRELOAD set to libcudaproxy.so
# The child process will invoke the CUDA library and use the GPU.
# The parent process interposes on all CUDA calls via libdmtcp_cuda.so,
#   and each wrapper around a CUDA call will communicate through a socket
#   to the child.  The child process (the "proxy") do the actual CUDA call.
# ISSUE:  The child process (proxy) is a copy of the user's application
#   so that it can invoke the CUDA kernels.  (A CUDA executable is a fat
#   binary with code for the host and for the device.)
#   But we must modify the application's main routine to enter a work loop
#   and process CUDA library calls sent to us by the original (parent) process.
#   libcudaproxy.so has that work loop.
#   In order to not require the user to modify their main routine
#   with the logic "if proxy, jump to work loop", we use LD_PRELOAD to
#   preload libcudaproxy.so.
#   Ideally, we would have simply defined our own 'main()' routine in
#   libcudaproxy.so and interpose on the user's 'main()'.  But this does
#   not work because CUDA's libcudart.so has its own constructor to
#   register the CUDA kernels of the fat binary.  LE_PRELOAD would cause
#   our own constructor to call our own main(), which never returns,
#   and so libcudart.so will never register the CUDA kernels of the fat binary.
#   So, our own 'main()' must be called after the libcudart.so constructor
#   and before the user's 'main().  We solve this problem with a trampoline
#   that causes the entry in main to jump to our own function,
#   libcudaproxy.so:main_wrapper().
# NOTE:
# libcudaproxy.so has a constructor so that it runs before 'main()'
# It creates a trampoline; then, at the call to main(), the thread of control
#   passes to libcudaproxy.so:main_wrapper() and never returns.
# To do this, libcudaproxy.so must evaluate '&main'.  The flag --export-dynamic
#   is needed so that the symbol 'main' is exported to libcudaproxy.so.
test/test%.o: test/test%.cu
	nvcc -cudart shared -g -O0 -ccbin ${NVCC_CXX} -c -o $@ $<
test/test%: test/test%.o
	g++ -export-dynamic -L$(NVCC_ROOT)/lib64 -lcudart -o $@ $<

# This recipe doesn't work.  Re-write it as above to make it work.
hpgmg: test/hpgmg.cu
	cd test; nvcc -cudart shared -g -O0 -std c++11 -ccbin ${NVCC_CXX} -c -o $@.o $@.cu
	cd test; g++ -L/usr/local/cuda-7.5/targets/x86_64-linux/lib/ -export-dynamic -o $@ $@.o -lcudart

# end: tests ------------------------------------------------


# These can be invoked as 'make check2.1', 'make check7', etc.
# We could delete all of these 'run' targets and use the 'check' targets.

run2.1: ${LIBNAME} libcudaproxy.so test2.1
	../../bin/dmtcp_launch -j  --with-plugin ./libdmtcp_cuda.so test/test2.1

run3: ${LIBNAME} libcudaproxy.so test3
		../../bin/dmtcp_launch -j  --with-plugin ./libdmtcp_cuda.so test/test3

run7: ${LIBNAME} libcudaproxy.so test7
	../../bin/dmtcp_launch -j  --with-plugin ./libdmtcp_cuda.so test/test7

run7.1: ${LIBNAME} libcudaproxy.so test7.1
	../../bin/dmtcp_launch -j  --with-plugin ./libdmtcp_cuda.so test/test7.1

restart:
	../../bin/dmtcp_restart -j ckpt*.dmtcp

# The dependencies of check could include everything in test subdirectory:
#   test1.cu test2.1.cu test2.cu test3.cu test4.cu test5.cu test6.cu
#   test7.1.cu test7.cu test10.cu
# 'make check7.1' will run 'text7.1.cu'
check: check1 check2

check%: test/test% ${LIBNAME} libcudaproxy.so 
	# Note that full path of plugin (using $$PWD in this case) is required.
	${DMTCP_ROOT}/bin/dmtcp_launch --port ${DEMO_PORT} \
	  --with-plugin $$PWD/${LIBNAME} $<

# We link the library using C++ for compatibility with the main libdmtcp.so
${LIBNAME}: ${LIBOBJS}
	${CXX} -cudart shared -shared -o $@ $^

cuda-wrappers.o: cuda-wrappers.cpp python-auto-generate/cudawrappers.icpp
	${CC} ${CFLAGS} -o $@ $<
.c.o:
	${CC} ${CFLAGS} -o $@ $<
.cpp.o:
	${CXX} ${CXXFLAGS} -o $@ $<

tidy:
	rm -f *~ .*.swp dmtcp_restart_script*.sh ckpt_*.dmtcp cudaSysCallsLog

clean:
	rm -f ${LIBOBJS} ${LIBNAME} test2 test/test2.1 \
	  libcudaproxy.so cudaSysCallsLog
	rm -f test/test_1 test/test1 test/test2 test/test3 test/test4 \
	  test/test5 test/test5 test/test6 test/test7
	rm -f libcudaproxy.so
	cd python-auto-generate && make clean

distclean: clean
	rm -f ${LIBNAME} *~ .*.swp dmtcp_restart_script*.sh ckpt_*.dmtcp

dist: distclean
	dir=`basename $$PWD`; cd ..; \
	  tar czvf $$dir.tar.gz --exclude-vcs ./$$dir
	dir=`basename $$PWD`; ls -l ../$$dir.tar.gz

.PHONY: default clean dist distclean
