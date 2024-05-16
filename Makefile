.PHONY: python cpp 

default: cpp

python: 
	cd Python && make && cd ..

cpp: 
	cd CPP && make && cd .. 
