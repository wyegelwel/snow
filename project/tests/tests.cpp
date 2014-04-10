/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   tests.cpp
**   Author: taparson
**   Created: 9 Apr 2014
**
**************************************************************************/

#include "tests.h"
#include <stdio.h>
#include <string.h>

//#include "cuda/testFunctions.h"
extern "C"
{
    void cumulativeSumTests();
    void groupParticlesTests();
    void weightingTestsHost();
    void testColliding();
}

void Tests::runTests(char *argv[])  {
    if (!strcmp(argv[2],"tim"))
    {
       runTimTests();
    }
    else if (!strcmp(argv[2],"eric"))
    {
        runEricTests();
    }
    else if (!strcmp(argv[2],"wil"))
    {
        runWilTests();
    }
    else if (!strcmp(argv[2], "all")){
        runTimTests();
        runEricTests();
        runWilTests();
    }
    //max, add your tests here as you like
    else
    {
        printf("Error: test name not found ...\n");
    }
}

void Tests::runTimTests()  {
    printf("running Tim Tests...\n");
    cumulativeSumTests();
    groupParticlesTests();
    printf("done running Tim Tests\n");
}

void Tests::runEricTests() {
    printf("running Eric Tests...\n");
    weightingTestsHost();
    printf("done running Eric Tests\n");
}

void Tests::runWilTests() {
    printf("running Wil Tests...\n");
    testColliding();
    printf("done running Wil Tests\n");
}
