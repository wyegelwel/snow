/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   selectiontool.h
**   Authors: evjang, mliberma, taparson, wyegelwe
**   Created: 20 Apr 2014
**
**************************************************************************/

#ifndef SELECTIONTOOL_H
#define SELECTIONTOOL_H

#include "ui/tools/tool.h"

class SceneNode;
struct vec3;

class SelectionTool : public Tool
{

public:

    SelectionTool( ViewPanel *panel,Type t);
    virtual ~SelectionTool();

    virtual void mousePressed();
    virtual void mouseReleased();

    virtual void update() {}

    virtual void render() {}

    bool hasSelection( vec3 &center ) const;
    bool hasRotatableSelection( vec3 &center ) const;
    bool hasScalableSelection( vec3 &center ) const;

    void clearSelection();
    SceneNode* getSelectedSceneNode();

};

#endif // SELECTIONTOOL_H
