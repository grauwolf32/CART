all:
	g++ -c cart.cpp -std=c++11
	g++ -c boost.cpp -std=c++11
	g++ -c leaf.cpp -std=c++11
	g++ -c utilites.cpp -std=c++11
	g++ main.cpp cart.o boost.o utilites.o leaf.o -std=c++11
	rm *.o
