add_subdirectory(gui)
add_subdirectory(utils)

add_executable(main_gui_node main_gui.cpp RosThread.cpp)
target_link_libraries(main_gui_node 
    gui 
    utils 
    glog
)
