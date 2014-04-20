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

    SelectionTool( ViewPanel *panel );
    virtual ~SelectionTool();

    virtual void mousePressed();
    virtual void mouseReleased();

    virtual void update() {}

    virtual void render() {}

    bool hasSelection() const;
    vec3 getSelectionCenter() const;

protected:

    void clearSelection();
    SceneNode* getSelectedSceneNode();

};

#endif // SELECTIONTOOL_H
