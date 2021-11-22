CC=mpicc

all:
	${CC} main.c -o main -fopenmp
run:
	mpirun -np 4 ./main
