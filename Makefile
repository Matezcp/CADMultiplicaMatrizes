CC=mpicc

all:
	${CC} main.c -o main -fopenmp
run:
	mpirun -np 4 ./main
	
Where -np is the number of process, do not need to be 4, it's just for demonstration
