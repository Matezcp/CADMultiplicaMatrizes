CC=mpicc
NP=4

all:
	${CC} main.c -o main -fopenmp
run:
	mpirun -np ${NP} ./main