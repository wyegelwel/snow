/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   tests.h
**   Author: taparson
**   Created: 9 Apr 2014
**
**************************************************************************/

#ifndef TESTS_H
#define TESTS_H

class Tests  {
public:
    static void runTests(char *argv[]);
private:
    static void runTimTests();
    static void runEricTests();
};

#endif // TESTS_H
