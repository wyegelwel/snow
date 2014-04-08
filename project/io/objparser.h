/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   objparser.h
**   Author: mliberma
**   Created: 8 Apr 2014
**
**************************************************************************/

#ifndef OBJPARSER_H
#define OBJPARSER_H

#include <QFile>
class Mesh;

class OBJParser
{

public:

    static bool load( const QString &filename, Mesh *mesh );

    OBJParser( const QString &filename, Mesh *mesh ) : m_file(filename), m_mesh(mesh) {}

    bool load();

private:

    QFile m_file;
    Mesh *m_mesh;

    bool parse( const QStringList &lines, int &lineIndex );

    bool parseVertex( const QString &line );
    bool parseFace( const QString &line );

};

#endif // OBJPARSER_H
