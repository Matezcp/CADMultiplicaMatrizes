CC=mpicc

all:
	${CC} main.c -o main -fopenmp
run:
	mpirun -np "número desejado de processos" ./main
