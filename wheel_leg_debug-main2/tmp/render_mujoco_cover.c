#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include <stdio.h>
#include <stdlib.h>

static void fail(const char* message) {
  fprintf(stderr, "%s\n", message);
  exit(1);
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s model.xml output.ppm\n", argv[0]);
    return 2;
  }

  if (!glfwInit()) fail("glfwInit failed");
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  glfwWindowHint(GLFW_SAMPLES, 4);
  GLFWwindow* window = glfwCreateWindow(1600, 1000, "MuJoCo offscreen", NULL, NULL);
  if (!window) fail("glfwCreateWindow failed");
  glfwMakeContextCurrent(window);

  char error[1024] = {0};
  mjModel* model = mj_loadXML(argv[1], NULL, error, sizeof(error));
  if (!model) fail(error);
  mjData* data = mj_makeData(model);
  mj_forward(model, data);

  mjvCamera camera;
  mjvOption option;
  mjvScene scene;
  mjrContext context;
  mjv_defaultCamera(&camera);
  mjv_defaultOption(&option);
  mjv_defaultScene(&scene);
  mjr_defaultContext(&context);

  camera.type = mjCAMERA_FREE;
  camera.lookat[0] = 0.0;
  camera.lookat[1] = 0.0;
  camera.lookat[2] = 0.48;
  camera.distance = 1.08;
  camera.azimuth = 135.0;
  camera.elevation = -18.0;

  mjv_makeScene(model, &scene, 3000);
  mjr_makeContext(model, &context, mjFONTSCALE_150);
  mjv_updateScene(model, data, &option, NULL, &camera, mjCAT_ALL, &scene);

  const int width = 1600;
  const int height = 1000;
  mjrRect viewport = {0, 0, width, height};
  mjr_render(viewport, &scene, &context);

  unsigned char* pixels = (unsigned char*)malloc((size_t)width * height * 3);
  if (!pixels) fail("pixel allocation failed");
  mjr_readPixels(pixels, NULL, viewport, &context);

  FILE* file = fopen(argv[2], "wb");
  if (!file) fail("could not open output file");
  fprintf(file, "P6\n%d %d\n255\n", width, height);
  for (int row = height - 1; row >= 0; --row) {
    fwrite(pixels + (size_t)row * width * 3, 1, (size_t)width * 3, file);
  }
  fclose(file);

  free(pixels);
  mjr_freeContext(&context);
  mjv_freeScene(&scene);
  mj_deleteData(data);
  mj_deleteModel(model);
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
