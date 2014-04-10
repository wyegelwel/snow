/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   objparser.cpp
**   Author: mliberma
**   Created: 8 Apr 2014
**
**************************************************************************/

#include <QString>
#include <QStringList>
#include <QRegExp>

#include "objparser.h"

#include "common/common.h"
#include "common/types.h"
#include "scene/mesh.h"

bool
OBJParser::load( const QString &filename, Mesh *mesh )
{
    OBJParser parser( filename, mesh );
    return parser.load();
}

bool
OBJParser::load()
{

    if ( m_mesh == NULL ) {
        LOG( "OBJParser has no mesh target!" );
        return false;
    }

    if ( !m_file.exists() || !m_file.open(QFile::ReadOnly) ) {
        LOG( "Unable to open file %s.", STR(m_file.fileName()) );
        return false;
    }

    QString text = QString(m_file.readAll());
    m_file.close();

    QStringList lines = text.split( QRegExp("[\r\n]"), QString::SkipEmptyParts );
    int lineIndex = 0;
    while ( lineIndex < lines.size() ) {
        if ( !parse(lines, lineIndex) ) {
            return false;
        }
    }

    if ( m_mesh->getNumNormals() != m_mesh->getNumVertices() )
        m_mesh->computeNormals();

    LOG( "Mesh loaded (%s)", STR(m_file.fileName()) );

    return true;

}

bool
OBJParser::parse( const QStringList &lines,
                  int &lineIndex )
{
    const QString &line = lines[lineIndex++];
    switch ( line[0].toLatin1() ) {
    case '#':
        break;
    case 'v':
        if ( !parseVertex(line) ) {
            LOG( "Error parsing vertex: %s", STR(line) );
            return false;
        }
        break;
    case 'f':
        if ( !parseFace(line) ) {
            LOG( "Error parsing face: %s", STR(line) );
            return false;
        }
        break;
    default:
        break;
    }

    return true;
}

bool
OBJParser::parseVertex( const QString &line )
{
    const static QRegExp regExp( "[\\s+v]" );
    QStringList str = line.split( regExp, QString::SkipEmptyParts );
    bool ok[3];
    m_mesh->addVertex( Vertex(str[0].toFloat(&ok[0]), str[1].toFloat(&ok[1]), str[2].toFloat(&ok[2])) );
    return ok[0] && ok[1] && ok[2];
}


// Parse face and break into triangles if necessary
bool
OBJParser::parseFace( const QString &line )
{
    const static QRegExp regExp( "[-\\s+f]" );
    QStringList str = line.split( regExp, QString::SkipEmptyParts );
    int nCorners = str.size();
    int *indices = new int[nCorners];
    for ( int i = 0; i < nCorners; ++i ) {
        bool ok;
        indices[i] = str[i].toInt(&ok)-1; // Note: OBJ indices start at 1
        if ( !ok ) return false;
    }
    int nTris = nCorners - 2;
    for ( int i = 0; i < nTris; ++i )
        m_mesh->addTri( Mesh::Tri(indices[0], indices[i+1], indices[i+2]) );
    delete [] indices;
    return true;
}
