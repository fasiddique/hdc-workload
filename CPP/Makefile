# Makefile for C++ project

# Compiler settings - Can change to clang++ if desired
CXX=g++
CXXFLAGS=-std=c++17 -Wall -g

# Linker settings
LDFLAGS=

# Find all cpp files in the current directory
SOURCES=$(wildcard *.cpp)
# Derive object file names from the source file names
OBJECTS=$(SOURCES:.cpp=.o)
# Define the executable name
EXECUTABLE=hdc-workload.out

# First rule is the one executed when no parameters are fed to the Makefile
all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^

# Rule for object files, each cpp file is compiled to an object file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Rule for cleaning up
clean:
	rm -f $(OBJECTS) $(EXECUTABLE) $(wildcard *.d)

# Rule for making everything afresh
rebuild: clean all

# Include dependencies
-include $(OBJECTS:.o=.d)

# Rule to generate a dependency file for each source file
%.d: %.cpp
	@$(CPP) $(CXXFLAGS) $< -MM -MT $(@:.d=.o) >$@
