/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   engine.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 14 Apr 2014
**
**************************************************************************/

#ifndef ENGINE_H
#define ENGINE_H

/**
 * @brief class that handles simulation and update of the ParticleSystem.
 * Simulates the ParticleSystem without drawing
 */

 /* In the future, the ViewPanel class will instantiate an engine to handle the simulation.
 * But for now I want to avoid breaking the GUI so will leave viewpanel.cpp alone until this is ready
 *
 */


#include <iostream>

class Engine
{
public:
    Engine();


    /**
     * loads simulation from XML file
     */
    void load(std::string fname);

    /**
     * Runs the simulation
     */
    bool start();

};

#endif // ENGINE_H
