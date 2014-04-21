#version 400 compatibility

in vec3 particlePosition;
in vec3 particleVelocity;
in float particleMass;
in float particleVolume;

out vec4 particleColor;

uniform int mode;
const int MASS = 0;
const int VELOCITY = 1;
const int SPEED = 2;

void main( void )
{
    particleColor = vec4( 1, 1, 1, 1 );
    if ( mode == MASS) {
        particleColor = vec4( 1, 1, 1, smoothstep(0.0, 1e-6, particleMass) );
    } else if ( mode == VELOCITY ) {
        particleColor = vec4( abs(particleVelocity), 1.f );
    } else if ( mode == SPEED ) {
        particleColor = mix( vec4(0.5, 0.5, 0.9, 1.0), vec4(0.9, 0.9, 0.9, 1.0), smoothstep(0.0, 5.0, length(particleVelocity)) );
    }
    gl_Position = gl_ModelViewProjectionMatrix * vec4( particlePosition, 1.0 );
    gl_PointSize = 1.0;
}
