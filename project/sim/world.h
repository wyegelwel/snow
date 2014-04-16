#ifndef WORLD_H
#define WORLD_H

struct WorldParams{
    float lambda; // first Lame parameter
    float mu; //second Lame paramter
    float xi; // Plastic hardening parameter
    float coeffFriction; // Coefficient of friction
};

#endif // WORLD_H
