/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   math.h
**   Author: mliberma
**   Created: 10 Apr 2014
**
**************************************************************************/

#ifndef MATH_H
#define MATH_H

#include <math.h>

#ifdef EPSILON
#undef EPSILON
#endif

#ifdef _EPSILON_
#undef _EPSILON_
#endif

#define _EPSILON_ 1e-6
#define EPSILON _EPSILON_

#define EQ(a, b) ( fabs((a) - (b)) < _EPSILON_ )
#define NEQ(a, b) ( fabs((a) - (b)) > _EPSILON_ )

#define EQF(a, b) ( fabsf((a) - (b)) < _EPSILON_ )
#define NEQF(a, b) ( fabsf((a) - (b)) > _EPSILON_ )

#ifdef MIN
#undef MIN
#endif

#define MIN( X, Y )                     \
({                                      \
    __typeof__ (X) _X_ = (X);           \
    __typeof__ (Y) _Y_ = (Y);           \
    ( (_X_<_Y_) ? _X_ : _Y_ );          \
})

#ifdef MAX
#undef MAX
#endif

#define MAX( X, Y )                     \
({                                      \
    __typeof__ (X) _X_ = (X);           \
    __typeof__ (Y) _Y_ = (Y);           \
    ( (_X_>_Y_) ? _X_ : _Y_ );          \
})


#endif // MATH_H
