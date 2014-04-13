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
#include <QString>
#include <QQueue>
#include <QVector>

#include "common/types.h"
#include "geometry/mesh.h"

class OBJParser
{

public:

    static void load( const QString &filename, QList<Mesh*> &meshes );

    OBJParser( const QString &filename = QString() ) : m_mode(VERTEX), m_file(filename), m_currentName("default") {}
    virtual ~OBJParser() { clear(); }

    inline QString getFileName() const { return m_file.fileName(); }
    inline void setFilename( const QString &filename ) { if ( m_file.isOpen() ) m_file.close(); m_file.setFileName(filename); }

    inline Mesh* popMesh() { return m_meshes.dequeue(); }
    inline bool hasMeshes() const { return !m_meshes.empty(); }

    bool load();
    void clear();

private:

    enum Mode { VERTEX, FACE, GROUP };

    Mode m_mode;

    QFile m_file;

    QString m_currentName;
    QVector<Vertex> m_vertexPool;
    QVector<Mesh::Tri> m_triPool;
    QVector<Normal> m_normalPool;

    QQueue<Mesh*> m_meshes;

    inline bool meshPending() const { return !m_vertexPool.empty() && !m_triPool.empty(); }
    void addMesh();

    void setMode( Mode mode );

    bool parse( const QStringList &lines, int &lineIndex );
    bool parseGroup( const QString &line );
    bool parseVertex( const QString &line );
    bool parseFace( const QString &line );

};

#endif // OBJPARSER_H
