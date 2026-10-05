cmake_minimum_required(VERSION 3.16)
project(rm_mujoco_controller C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/ThirdParty.cmake")

find_package(OpenGL REQUIRED)

set(CONTROL_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")

add_library(rm_controller STATIC
    "${CONTROL_ROOT}/Application/Task/Src/Chassis_Task.c"
    "${CONTROL_ROOT}/Application/Task/Src/ChassisL_Task.c"
    "${CONTROL_ROOT}/Application/Task/Src/ChassisR_Task.c"
    "${CONTROL_ROOT}/Components/Algorithm/Src/VMC_Calc.c"
    "${CONTROL_ROOT}/Components/Algorithm/Src/LPF.c"
    "${CONTROL_ROOT}/Components/Controller/Src/PID.c"
    "${CONTROL_ROOT}/Components/Device/Src/Motor.c"
    "${CONTROL_ROOT}/Components/Lib/Src/User_Lib.c"
    "${CMAKE_CURRENT_LIST_DIR}/jump_state_machine.c"
    "${CMAKE_CURRENT_LIST_DIR}/sim_adapter.c"
    "${CMAKE_CURRENT_LIST_DIR}/port/sim_platform.c"
)

target_compile_definitions(rm_controller PRIVATE SIM_MUJOCO)

target_include_directories(rm_controller PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}"
    "${CMAKE_CURRENT_LIST_DIR}/port"
    "${CONTROL_ROOT}/Application/Task/Inc"
    "${CONTROL_ROOT}/Application/RobotParam/Inc"
    "${CONTROL_ROOT}/Application/TypeDef/Inc"
    "${CONTROL_ROOT}/Components/Algorithm/Inc"
    "${CONTROL_ROOT}/Components/Controller/Inc"
    "${CONTROL_ROOT}/Components/Device/Inc"
    "${CONTROL_ROOT}/Components/Lib/Inc"
)

target_link_libraries(rm_controller PUBLIC m)

add_executable(mujoco_bridge
    "${CMAKE_CURRENT_LIST_DIR}/main_mujoco.c"
    "${CMAKE_CURRENT_LIST_DIR}/steer_state_machine.c"
    "${CMAKE_CURRENT_LIST_DIR}/jump_telemetry.c"
)

target_compile_definitions(mujoco_bridge PRIVATE SIM_MUJOCO)
target_link_libraries(mujoco_bridge PRIVATE rm_controller m OpenGL::GL)
rm_target_link_glfw(mujoco_bridge)
rm_target_link_mujoco(mujoco_bridge)
