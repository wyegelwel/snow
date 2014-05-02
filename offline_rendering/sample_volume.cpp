#include <iomanip>
#include <fstream>
#include <stdio.h>
#include "math.h"

void write_data(int frame)
{
    int xres=128;
    int yres=128;
    int zres=128;

    float t = float(frame)/24;

    char fname[128];
    snprintf(fname, sizeof(fname), "/data/people/evjang/offline_renders/mts_scene/test_%04i.vol", frame);
    std::ofstream os(fname);

    os.write("VOL",3);
    char version  = 3;
    os.write((char*)&version,sizeof(char));
    int value=1;
    os.write((char*)&value,sizeof(int));
    os.write((char*)&xres,sizeof(int));
    os.write((char*)&yres,sizeof(int));
    os.write((char*)&zres,sizeof(int));

    int channels=1;
    os.write((char*)&channels,sizeof(int));


    float minX=-.5;
    float maxX=.5;
    
    float minY=0;
    float maxY=1;

    float minZ=-.5;
    float maxZ=.5;

    os.write((char *) &minX, sizeof(float));
    os.write((char *) &minY, sizeof(float));
    os.write((char *) &minZ, sizeof(float));
    os.write((char *) &maxX, sizeof(float));
    os.write((char *) &maxY, sizeof(float));
    os.write((char *) &maxZ, sizeof(float));

    // data[((zpos*yres + ypos)*xres + xpos)*channels + chan]
    float r2 = .2 * .2;
    float x,y,z;
    float sx,sy,sz;
    float dx,dy,dz;
    float h = float(maxX-minX)/xres;
int success = 0;
    printf("sphere center : %f, %f, %f \n", sx, sy, sz);
    for (int i=0; i<xres*yres*zres; ++i)
    {
        sx = (maxX+minX)/2;
        sy = (maxY+minY)/2 + .5*sin(t*20); // center of simulation box
        sz = (maxZ+minZ)/2;

        int j=i;
        int xpos = j%xres;
        j = (j-xpos)/xres;
        int ypos = j%yres;
        j = (j-ypos)/yres;
        int zpos = j;

        x = minX + xpos*h;
        y = minY + ypos*h;
        z = minZ + zpos*h;

       // printf("%f %f %f \n", x,y,z);

        dx = x-sx;
        dy = y-sy;
        dz = z-sz;

        //float density = (dx*dx + dy*dy + dz*dz < r2) ? 1.f : 0.f;
        //float density = sin(z*20);
        float density = 0;
        
        if (dx*dx + dy*dy + dz*dz < r2)
        {
            success++;
            //printf("success\n");
            density = 1;
        }


        os.write((char*)&density,sizeof(float));
    }   
    printf("num success: %d \n", success);
}

int main()
{
    float seconds = 2;
    write_data(0);
	// for (int f=0; f<int(seconds*24); f++)
 //    {
 //        write_data(f);
 //    }
    printf("done \n ");
	return 0;
}
