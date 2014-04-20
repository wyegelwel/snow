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

void
OBJParser::load( const QString &filename, QList<Mesh *> &meshes )
{
    OBJParser parser( filename );
    if ( parser.load() ) {
        while ( parser.hasMeshes() ) {
            meshes += parser.popMesh();
        }
    }
}

void
OBJParser::clear()
{
    while ( !m_meshes.empty() ) {
        Mesh* mesh = m_meshes.dequeue();
        SAFE_DELETE( mesh );
    }
    m_vertexPool.clear();
    m_normalPool.clear();
}

bool
OBJParser::load()
{

    clear();

    if ( m_file.fileName().isEmpty() ) {
        LOG( "OBJParser: No file name!" );
        return false;
    }

    if ( !m_file.exists() || !m_file.open(QFile::ReadOnly) ) {
        LOG( "OBJParser: Unable to open file %s.", STR(m_file.fileName()) );
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

    if ( meshPending() ) addMesh();

    for ( QQueue<Mesh*>::iterator it = m_meshes.begin(); it != m_meshes.end(); ++it ) {
        if ( (*it)->getNumNormals() != (*it)->getNumVertices() ) {
            (*it)->computeNormals();
        }
    }

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
    case 'g':
        if ( !parseGroup(line) ) {
            LOG( "Error parsing group: %s", STR(line) );
            return false;
        }
        break;
    case 'v':
        switch ( line[1].toLatin1() ) {
        case ' ': case'\t':
            if ( !parseVertex(line) ) {
                LOG( "Error parsing vertex: %s", STR(line) );
                return false;
            }
            break;
        default:
            break;
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
OBJParser::parseGroup( const QString &line )
{
    setMode( GROUP );
    const static QRegExp regExp( "[\\s+\n\r]" );
    QStringList lineStrs = line.split( regExp, QString::SkipEmptyParts );
    m_currentName = ( lineStrs.size() > 1 ) ? lineStrs[1] : "default";
    return true;
}

bool
OBJParser::parseVertex( const QString &line )
{
    setMode( VERTEX );
    const static QRegExp regExp( "[\\s+v]" );
    QStringList vertexStrs = line.split( regExp, QString::SkipEmptyParts );
    bool ok[3];
    m_vertexPool += Vertex(vertexStrs[0].toFloat(&ok[0]), vertexStrs[1].toFloat(&ok[1]), vertexStrs[2].toFloat(&ok[2]));
    return ok[0] && ok[1] && ok[2];
}


// Parse face and break into triangles if necessary
bool
OBJParser::parseFace( const QString &line )
{
    setMode( FACE );
    const static QRegExp regExp( "[-\\s+f]" );
    QStringList faceStrs = line.split( regExp, QString::SkipEmptyParts );
    int nCorners = faceStrs.size();
    int *indices = new int[nCorners];
    for ( int i = 0; i < nCorners; ++i ) {
        const static QRegExp regExp2( "[/]" );
        QStringList cornerStrs = faceStrs[i].split( regExp2, QString::KeepEmptyParts );
        bool ok;
        indices[i] = cornerStrs[0].toInt(&ok)-1; // Note: OBJ indices start at 1
        if ( !ok ) return false;
    }
    int nTris = nCorners - 2;
    for ( int i = 0; i < nTris; ++i )
        m_triPool += Mesh::Tri(indices[0], indices[i+1], indices[i+2]);
    delete [] indices;
    return true;
}

void
OBJParser::setMode( Mode mode )
{
    if ( mode != m_mode ) {
        if ( mode == VERTEX ) {
            addMesh();
        }
        m_mode = mode;
    }
}

void
OBJParser::addMesh()
{
    if ( meshPending() ) {
        LOG( "OBJParser: adding mesh %s...", STR(m_currentName) );
        Mesh *mesh = new Mesh;
        mesh->setName( m_currentName );
        mesh->setFilename( m_file.fileName() );
        mesh->setVertices( m_vertexPool );
        mesh->setTris( m_triPool );
        if ( m_normalPool.size() == m_vertexPool.size() )
            mesh->setNormals( m_normalPool );
        m_meshes.enqueue( mesh );
        m_currentName = "default";
        m_vertexPool.clear();
        m_triPool.clear();
        m_normalPool.clear();
    }
}
