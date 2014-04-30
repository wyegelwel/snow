#version 400 compatibility

uniform vec3 pos;
uniform vec3 dim;
uniform float h;
uniform float density;

in float nodeMass;
in vec3 nodeVelocity;
in vec3 nodeForce;

out vec4 nodeColor;

uniform int mode;
const int MASS = 0;
const int VELOCITY = 1;
const int SPEED = 2;
const int FORCE = 3;

void main( void )
{
    float alpha = 0.75 * smoothstep( 0.0, density, nodeMass/(h*h*h) );
    nodeColor = vec4( 0.8, 0.8, 0.9, alpha );
    if ( mode == VELOCITY ) {
        nodeColor.rgb = abs(nodeVelocity);
    } else if ( mode == SPEED ) {
        float speed = smoothstep( 0.0, 5.0, length(nodeVelocity) );
        nodeColor.rgb = mix( vec3(0.15, 0.15, 0.9), nodeColor.rgb, speed );
    } else if ( mode == FORCE ) {
        nodeColor.rgb = 1000.0*abs(nodeForce);
    }
    float i = gl_VertexID;
    float x = floor(i/((dim.y+1)*(dim.z+1)));
    i -= x*(dim.y+1)*(dim.z+1);
    float y = floor(i/(dim.z+1));
    i -= y*(dim.z+1);
    float z = i;
    vec4 position = vec4( pos + h * vec3( x, y, z ), 1.0 );
    gl_Position = gl_ModelViewProjectionMatrix * position;
    gl_PointSize = 3.0;
}
