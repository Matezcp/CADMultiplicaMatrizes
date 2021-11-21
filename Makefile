CC=mpicc

all:
	${CC} main.c -o main -fopenmp
run:
	./main