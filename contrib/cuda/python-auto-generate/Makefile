# Makefile for *.h, *.icu, *.icpp

check: cuda_plugin.h cudaproxy.icu cudawrappers.icpp
	@ echo '*** CREATED:' $^

check-%: %-cuda_plugin.h %-cudaproxy.icu %-cudawrappers.icpp
	@ echo '*** CREATED:' $^

%-cuda_plugin.h %-cudaproxy.icu %-cudawrappers.icpp: cuda-wrappers.py %.template
	python $^
cuda_plugin.h cudaproxy.icu cudawrappers.icpp: cuda-wrappers.py main.template
	python $^

touch: cuda-wrappers.py
	touch $<
vi: cuda-wrappers.py
	vim $<

dist: clean
	dir=`basename $$PWD`&& cd ..&& tar czvf $$dir.tgz ./$$dir
	dir=`basename $$PWD`&& ls -l ../$$dir.tgz

clean:
	rm -f *cuda_plugin.h *.icu *.icpp

.PHONY: vi vim clean check-%
