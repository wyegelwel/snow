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
//#include "cuda/testFunctions.h"
extern "C"
void cumulativeSumTests();


void Tests::runTests()  {
    runTimTests();
}

void Tests::runTimTests()  {
    printf("running Tim Tests...\n");
    cumulativeSumTests();
    printf("done running Tim Tests\n");
}
