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



