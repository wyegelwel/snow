/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   common.h
**   Author: mliberma
**   Created: 6 Apr 2014
**
**************************************************************************/

#ifndef COMMON_H
#define COMMON_H

#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>

#define SAFE_DELETE(MEM)                \
    {                                   \
        if ((MEM)) {                    \
            delete ((MEM));             \
            (MEM) = NULL;               \
        }                               \
    }

#define SAFE_DELETE_ARRAY(MEM)          \
    {                                   \
        if ((MEM)) {                    \
            delete[] ((MEM));           \
            (MEM) = NULL;               \
        }                               \
    }

#ifndef QT_NO_DEBUG
    #define LOG(...) {                                  \
        time_t rawtime;                                 \
        struct tm *timeinfo;                            \
        char buffer[16];                                \
        time( &rawtime );                               \
        timeinfo = localtime( &rawtime );               \
        strftime( buffer, 16, "%H:%M:%S", timeinfo );   \
        fprintf( stderr, "[%s] ", buffer );             \
        fprintf( stderr, __VA_ARGS__ );                 \
        fprintf( stderr, "\n" );                        \
        fflush( stderr );                               \
    }
    #define LOGIF( TEST, ... ) {                            \
        if ( (TEST) ) {                                     \
            time_t rawtime;                                 \
            struct tm *timeinfo;                            \
            char buffer[16];                                \
            time( &rawtime );                               \
            timeinfo = localtime( &rawtime );               \
            strftime( buffer, 16, "%H:%M:%S", timeinfo );   \
            fprintf( stderr, "[%s] ", buffer );             \
            fprintf( stderr, __VA_ARGS__ );                 \
            fprintf( stderr, "\n" );                        \
            fflush( stderr );                               \
        }                                                   \
    }
    #define TIME( START, END, ... ) {                       \
        timeval start, end;                                 \
        gettimeofday( &start, NULL );                       \
        printf( "%s", START ); fflush(stdout);              \
        { __VA_ARGS__ };                                    \
        gettimeofday( &end, NULL );                         \
        long int ms = (end.tv_sec-start.tv_sec)*1000 +      \
                  (end.tv_usec-start.tv_usec)/1000;         \
        printf( "[%ld ms] %s", ms, END ); fflush(stdout);   \
    }
#else
    #define LOG(...) do {} while(0)
    #define LOGIF( TEST, ... ) do {} while(0)
    #define TIME( START, END, ... ) do {} while(0)
#endif

#define STR( QSTR ) QSTR.toStdString().c_str()

#endif // COMMON_H
