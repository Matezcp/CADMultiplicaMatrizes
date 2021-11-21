#include<string.h>
#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include <omp.h>
#include"mpi.h"

#define TAG 13
//Matrix N X N
#define N 4
//Número de Threads para paralelização
#define NUMTHREADS 8

int main(int argc, char **argv){
    int **A,**Aoriginal,**B,**C,**Cfinal,*tmp;
    int numProcs,myRank,cargaTrabalho;
    int numThreads = NUMTHREADS;
    int i,j,k;
    MPI_Comm USED_PROCESS;

    //Inicializa o MPI e pega o número de processos e o rank do processo atual
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  
    //Seta o número de threads do OMP
    omp_set_num_threads(numThreads);

    //Define o quanto cada processo irá receber
    if(numProcs >= N)
        cargaTrabalho = 1;
    else
        cargaTrabalho = N/numProcs;

    //Coloca os processos que serão usados em um sub-comunicador e os processo que não irão ser utilizados em outro
    MPI_Comm_split(MPI_COMM_WORLD, (myRank<N), myRank, &USED_PROCESS);

    //Processos alocam as matrizes que irão utilizar nos calculos
    if(myRank < N){
        //Aloca a matriz A
        tmp = (int *) malloc (sizeof(int) * N * cargaTrabalho);
        A = (int **) malloc(sizeof(int*) * cargaTrabalho);
        for(i = 0; i < cargaTrabalho; i++){
            A[i] = &tmp[i * N];
        }
        //Aloca a matriz B
        tmp = (int *) malloc (sizeof(int) * N * N);
        B = (int **) malloc(sizeof(int*) * N);
        for(i = 0; i < N; i++){
            B[i] = &tmp[i * N];
        }
        //Aloca a matriz C
        tmp = (int *) malloc (sizeof(int) * N * cargaTrabalho);
        C = (int **) malloc(sizeof(int*) * cargaTrabalho);
        for(i = 0; i < cargaTrabalho; i++){
            C[i] = &tmp[i * N];
        }
    }

    //Processo 0 aloca as matrizes originais (a matriz B já foi alocada)
    if(myRank == 0){
        //Aloca a matriz A original
        tmp = (int *) malloc (sizeof(int) * N * N);
        Aoriginal = (int **) malloc(sizeof(int*) * N);
        for(i = 0; i < N; i++){
            Aoriginal[i] = &tmp[i * N];
        }
        //Aloca a matriz C Final
        tmp = (int *) malloc (sizeof(int) * N * N);
        Cfinal = (int **) malloc(sizeof(int*) * N);
        for(i = 0; i < N; i++){
            Cfinal[i] = &tmp[i * N];
        }
        //Preenche Aoriginal e B (NÃO TÁ ALEÁTORIO LÁ DO JEITO QUE TEM SER)
        k = 0;
        for(i = 0; i < N; i++){
            for(j = 0; j < N; j++){
                Aoriginal[i][j] = k;
                B[i][j] = k;
                k++;
            }
        }
    }

    
    if(myRank < N){
        //Manda um pedaço da matriz Aoriginal equivalente a carga de trabalho para cada processo
        if(MPI_Scatter(Aoriginal[0], N*cargaTrabalho, MPI_INT, A[0], N*cargaTrabalho, MPI_INT, 0, USED_PROCESS) != MPI_SUCCESS){
            printf("SCATTER ERROR\n");
            exit(1);
        }
        //Manda a matrix B inteira para todos processos
        MPI_Bcast(B[0],N*N,MPI_INT,0,USED_PROCESS);

        //Inicializa a matriz C
        for(i = 0; i < cargaTrabalho; i++){
            for(j = 0; j < N; j++){
                C[i][j] = 0;
            }
        }

        //Cada processo faz seus calculos paralelamente
        //#pragma omp parallel for shared(A,B,C,numThreads) private(i,j,k) schedule (static, 10)
        for (i = 0; i < cargaTrabalho; i++) {
            for (j = 0; j < N; j++) {
                for (k = 0; k < N; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        //O processo 0 recebe as partes de cada processo
        if(MPI_Gather(C[0], N*cargaTrabalho, MPI_INT, Cfinal[0], N*cargaTrabalho, MPI_INT, 0, USED_PROCESS) != MPI_SUCCESS){
            printf("GATHER ERROR\n");
            exit(1);
        }

        //Printa o resultado Final
        if(myRank == 0){
            for(i = 0; i < N; i++){
                for(j = 0; j < N; j++){
                    printf("%d ",Cfinal[i][j]);
                }
                printf("\n");
            }
        }
    }

    //Espera todos processos chegarem aqui antes de finalizar as execuções
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}