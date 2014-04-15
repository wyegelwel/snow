/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   main_console.cpp
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 15 Apr 2014
**
**************************************************************************/


#include <iostream>
#include <iomanip>
#include "sim/engine.h"
#include <string.h>

void printHelp()
{
    std::cout << "Usage : ./snow_console -o [OUTPUT_DIR] [SCENE.XML]" << std::endl;
    std::cout << "Runs a snow simulation without GUI\n" << std::endl;

}

int main(int argc, char *argv[])
{
    if (argc == 5 && !strcmp(argv[1],"-sim"))  {
        /**
         * runs an instance of engine and runs the particle simulation on it.
         *
         */
        Engine * e = new Engine();
        std::string outputdir(argv[3]);
        std::string scenefile(argv[4]);
        e->load(scenefile);
        // run the simulation
        e->start();
        delete e;
    }
    else if (argc == 2 && (!strcmp(argv[1],"-h") || !strcmp(argv[1],"--help")))
    {
        printHelp();
    }
    else
    {
        std::cout << "invalid option" << std::endl;
        printHelp();
    }

    return 0;
}
