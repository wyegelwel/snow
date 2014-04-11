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
