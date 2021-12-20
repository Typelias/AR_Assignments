#ifdef GL_ES
    precision highp float;
#endif

uniform vec3 color = vec3 (0.5,0.5,0.5);
uniform vec3 sunPosition = vec3(0,2, 0);
uniform float ambientLightStreanght = 0.2;
uniform float lambertianLightStreanght = 0.7;
varying vec3 normal;

void main (void){
    vec3 lambertian = max(dot(normal, sunPosition), 0.0);
    vec3 finalIntensity = (lambertian + ambientLightStreanght) * color;
    gl_FragColor = vec4(finalIntensity, 1.0);
}