#include <stdio.h>
#include <omp.h>
#include <unistd.h>

int main(int argc,char *argv[]) {

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i=0; i<20; i++)
            {
                #pragma omp task
                sleep(2);
            }
            printf("Hello from me: %i\n", omp_get_thread_num());
            #pragma omp taskwait
        }
    }
    return 0;
}