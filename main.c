/*
GRUPO:
Mateus Zanetti Camargo Penteado - 11219202
Yure Pablo do Nascimento Oliveira - 11275317
Breno Alves de Sousa - 11345555
*/
#include<string.h>
#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include <omp.h>
#include"mpi.h"

//Matrix N X N
#define N 10
//Número de Threads para paralelização
#define NUMTHREADS 8
//Para debug
//#define DEBUG

int main(int argc, char **argv){
    long long int *A,*Aoriginal,*B,*C,*Cfinal;
    int numProcs,myRank,cargaTrabalho;
    long long int numThreads = NUMTHREADS;
    long long int i,j,k;
    double wtime;
    MPI_Comm USED_PROCESS;

    //Inicializa o MPI e pega o número de processos e o rank do processo atual
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    //Define o quanto cada processo irá receber
    if(numProcs >= N)
        cargaTrabalho = 1;
    else
        cargaTrabalho = N/numProcs;
    
    //Seta o número de threads do OMP
    omp_set_num_threads(numThreads);
    
    //Seta a seed para o rand
    srand(2021);

    //Coloca os processos que serão usados em um sub-comunicador com cor 1 e os processo que não irão ser utilizados em outro com cor 0
    MPI_Comm_split(MPI_COMM_WORLD, (myRank<N), myRank, &USED_PROCESS);

    //Processos alocam as matrizes que irão utilizar nos calculos (as matrizes são alocadas como vetores, pois precisamos que os dados
    //fiquem sequenciais na memória)
    if(myRank < N){
        //Aloca a matriz A
        A = (long long int *) malloc(sizeof(long long int) * N * cargaTrabalho);
        //Aloca a matriz B
        B = (long long int *) malloc(sizeof(long long int) * N * N);
        //Aloca a matriz C
        C = (long long int *) calloc(N * cargaTrabalho,sizeof(long long int));
    }

    //Processo 0 aloca as matrizes originais (a matriz B já foi alocada)
    if(myRank == 0){
        //Aloca a matriz A original
        Aoriginal = (long long int *) malloc (sizeof(long long int) * N * N);
        //Aloca a matriz C Final
        Cfinal = (long long int *) calloc (N*N,sizeof(long long int));

        //Preenche Aoriginal e B
        for(i = 0; i < N; i++){
            for(j = 0; j < N; j++){
                //Limitamos a 50 os valores das matrizes
                Aoriginal[i*N+j] = rand() % 50;
                B[i*N+j] = rand() % 50;
            }
        }

        #ifdef DEBUG
        printf("Aoriginal: \n");
        for(i = 0; i < N; i++){
            for(j = 0; j < N; j++){
                printf("%lld ",Aoriginal[i*N+j]);
            }
            printf("\n");
        }

        printf("B: \n");
        for(i = 0; i < N; i++){
            for(j = 0; j < N; j++){
                printf("%lld ",B[i*N+j]);
            }
            printf("\n");
        }

        printf("\n");
        #endif
    }

    
    if(myRank < N){
        //Manda um pedaço da matriz Aoriginal equivalente a carga de trabalho para cada processo
        if(MPI_Scatter(Aoriginal, N*cargaTrabalho, MPI_LONG_LONG_INT, A, N*cargaTrabalho, MPI_LONG_LONG_INT, 0, USED_PROCESS) != MPI_SUCCESS){
            printf("SCATTER ERROR\n");
            exit(1);
        }
        //Manda a matrix B inteira para todos processos
        MPI_Bcast(B,N*N,MPI_LONG_LONG_INT,0,USED_PROCESS);

        //Começa o timer
        if (myRank == 0) {
            wtime = omp_get_wtime();
        }

        //Cada processo faz seus calculos paralelamente, primeiro colapsamos os 2 primeiros for's
        #pragma omp parallel for collapse(2)
        for (i = 0; i < cargaTrabalho; i++) {
            for (j = 0; j < N; j++) {
                long long int soma = 0;

                #ifdef DEBUG
                    printf("On thread %d, i = %lld, j = %lld\n", omp_get_thread_num(), i, j);
                #endif
                //Em seguida paralelizamos o for mais interno com simd e redução da variável soma
                #pragma omp parallel for simd reduction(+:soma)
                for (k = 0; k < N; k++) {
                    soma += A[i*N+k] * B[k*N+j];
                }
                //Atualizamos o valor de C calculado
                C[i*N+j] = soma;
            }
        }

        if(myRank == 0){
            //Verifica se a carga de trabalho não foi exata
            if(numProcs*cargaTrabalho != N){
                //Ve o quanto falta computar
                long long int faltantes = N - numProcs*cargaTrabalho;
                //Computa o que falta, primeiro colapsamos os 2 primeiros for's
                #pragma omp parallel for collapse(2)
                for (i = N-1; i >= N-faltantes; i--) {
                    for (j = 0; j < N; j++) {
                        long long int soma = 0;
                        //Em seguida paralelizamos o for mais interno com simd e redução da variável soma
                        #pragma omp parallel for simd reduction(+:soma)
                        for (k = 0; k < N; k++) {
                            soma += Aoriginal[i*N+k] * B[k*N+j];
                        }
                        //Atualizamos o valor de C calculado
                        Cfinal[i*N+j] = soma;
                    }
                }
            }
        }

        //O processo 0 recebe as partes de cada processo
        if(MPI_Gather(C, N*cargaTrabalho, MPI_LONG_LONG_INT, Cfinal, N*cargaTrabalho, MPI_LONG_LONG_INT, 0, USED_PROCESS) != MPI_SUCCESS){
            printf("GATHER ERROR\n");
            exit(1);
        }

        //Termina o timer
        if(myRank == 0){
            wtime = omp_get_wtime() - wtime;
        }

        //Printa o resultado Final
        if(myRank == 0){
            for(i = 0; i < N; i++){
                for(j = 0; j < N; j++){
                   printf("%lld ",Cfinal[i*N+j]);
                }
                printf("\n");
            }
            printf("Tempo passado: %lf\n", wtime);
        }
    }

    //Espera todos processos chegarem aqui antes de finalizar as execuções
    MPI_Barrier(MPI_COMM_WORLD);
    //Libera o que foi alocado
    if(myRank < N){
        free(A);
        free(B);
        free(C);
        if(myRank == 0){
            free(Aoriginal);
            free(Cfinal);
        }
    }
    MPI_Finalize();
    return 0;
}