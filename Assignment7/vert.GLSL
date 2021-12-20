#ifdef GL_ES
    precision highp float;
#endif

attribute vec4 vertexPosition;
attribute vec4 normalPosition;
uniform mat4 model_matrix, view_matrix, projection_matrix, normal_matrix;
varying vec3 normal; 

void main() {
  normal = normalize(normal_matrix * (normalPosition)).xyz;
  gl_Position = projection_matrix * view_matrix * model_matrix * vertexPosition; 
}