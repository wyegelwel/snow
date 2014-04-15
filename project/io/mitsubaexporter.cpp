/**************************************************************************
**
**   SNOW - CS224 BROWN UNIVERSITY
**
**   mitsubaexporter.cpp
**   Author: evjang
**   Created: 13 Apr 2014
**
**************************************************************************/

#include "mitsubaexporter.h"

MitsubaExporter::MitsubaExporter()
{
}

void MitsubaExporter::exportPointCloud(QVector<Particle> particles, BBox bounds)
{
    // std::string path = dir +
        //std::ofstream o();
       // Float scale = 1.0f / std::max(std::max(xres, yres), zres);

    //        os.write("VOL", 3);
    //        char version = 3;
    //        os.write((char *) &version, sizeof(char));
    //        int value = 1;
    //        os.write((char *) &value, sizeof(int));
    //        os.write((char *) &xres, sizeof(int));
    //        os.write((char *) &yres, sizeof(int));
    //        os.write((char *) &zres, sizeof(int));
    //        value = 1;
    //        os.write((char *) &value, sizeof(int));

    //        float minX = -xres/2.0f*scale;
    //        float minY = -yres/2.0f*scale;
    //        float minZ = -zres/2.0f*scale;
    //        float maxX = xres/2.0f*scale;
    //        float maxY = yres/2.0f*scale;
    //        float maxZ = zres/2.0f*scale;

    //        os.write((char *) &minX, sizeof(float));
    //        os.write((char *) &minY, sizeof(float));
    //        os.write((char *) &minZ, sizeof(float));
    //        os.write((char *) &maxX, sizeof(float));
    //        os.write((char *) &maxY, sizeof(float));
    //        os.write((char *) &maxZ, sizeof(float));
    //        for (size_t i=0; i<density.size(); ++i) {
    //            float value = density[i].density;
    //            os.write((char *) &value, sizeof(float));
    //        }
    //        os.close();
}

void MitsubaExporter::exportScene(std::string dir, int frame, SceneNode *sceneNode)
{
    //
}
