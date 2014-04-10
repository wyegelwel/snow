/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   main.cpp
**   Author: mliberma
**   Created: 6 Apr 2014
**
**************************************************************************/


#define devMode true

#if devMode == true

#include <QApplication>
#include "ui/mainwindow.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    
    return a.exec();
}


#else

extern "C"
void runTestsEric();

int main(int argc, char *argv[])
{
    std::cout << "hello world!" << std::endl;
}


#endif
