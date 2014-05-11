#version 400 compatibility

in vec3 particlePosition;
in vec3 particleVelocity;
in float particleMass;
in float particleVolume;
in float particleStiffness; // mat.xi

out vec4 particleColor;

uniform int mode;
const int MASS = 0;
const int VELOCITY = 1;
const int SPEED = 2;
const int STIFFNESS = 3;

void main( void )
{
    particleColor = vec4( 0.8, 0.8, 0.9, 1.0 );
    if ( mode == MASS) {
        particleColor = vec4( 1.0, 1.0, 1.0, 1.0 );
    } else if ( mode == VELOCITY ) {
        particleColor = vec4( abs(particleVelocity), 1.0 );
    } else if ( mode == SPEED ) {
        particleColor = mix( vec4(0.15, 0.15, 0.9, 1.0), vec4(0.9, 0.9, 0.9, 1.0), smoothstep(0.0, 5.0, length(particleVelocity)) );
    } else if ( mode == STIFFNESS ) {
        float n = (particleStiffness - 5.f)/5.f;
        particleColor = vec4(vec3(n),1);
    }

    gl_Position = gl_ModelViewProjectionMatrix * vec4( particlePosition, 1.0 );
    gl_PointSize = 3.0;
}
