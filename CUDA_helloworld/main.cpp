#include <QCoreApplication>

extern "C" // tell compiler that function is defined somewhere else

void runCudaPart();

#include <stdio.h>

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    runCudaPart();
    return a.exec();
}


//__global__ void helloCUDA(float f)
//{
//    printf("Hello thread %d, f=%f\n", threadIdx.x, f);
//}

//int main()
//{
//    helloCUDA<<<1, 5>>>(1.2345f);
//    cudaDeviceSynchronize();
//    return 0;
//}
