# # Makefile for C++ project


# # Compiler settings - Can change to clang++ if desired
# CXX=g++
# CXXFLAGS = -std=c++17 -Wall -I../../../libpimsim/include -L../../../libpimsim/lib -lpimsim

# all: $(EXECUTABLE)

# # Linker settings
# test: 
# 	echo $(CXXFLAGS)
# # Setup libpimsim library
# LDFLAGS=

# # Find all cpp files in the current directory
# SOURCES=$(wildcard *.cpp)

# # Derive object file names from the source file names and place them in the obj directory
# OBJECTS=$(patsubst %.cpp, obj/%.o, $(SOURCES))

# # Define the executable name
# EXECUTABLE=hdc-workload.out



# $(EXECUTABLE): $(OBJECTS)
# 	$(CXX)  $(CXXFLAGS) -o $@ $^

# # Create the obj directory if it doesn't exist
# obj/%.o: %.cpp | obj
# 	$(CXX) $(CXXFLAGS) -c -o $@ $<

# # Rule for creating the obj directory
# obj:
# 	mkdir -p obj

# # Rule for cleaning up
# clean:
# 	rm -f $(OBJECTS) $(EXECUTABLE) $(wildcard *.d)
# 	rm -rf obj

# # Rule for making everything afresh
# rebuild: clean all

# # Include dependencies
# -include $(patsubst %.cpp, obj/%.d, $(SOURCES))

# # Rule to generate a dependency file for each source file
# obj/%.d: %.cpp | obj
# 	@$(CPP) $(CXXFLAGS) $< -MM -MT $(patsubst %.cpp, obj/%.o, $<) >$@
all: 
	g++  -std=c++17 -Wall -I../../../libpimsim/include -L../../../libpimsim/lib -lpimsim -o hdc-workload.out *.cpp ../../../libpimsim/lib/libpimsim.a