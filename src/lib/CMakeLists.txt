
MACRO(make_lib name)
    #Bring the sources
    FILE(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/lib/${name}/*.cpp)

    #Generate library
    ADD_LIBRARY(${name} SHARED ${SOURCES})

    #Add other libraries used to build the main library
    TARGET_LINK_LIBRARIES(${name} PRIVATE ${OpenCV_LIBS})

    #Set an installation path, type "sudo make install" to apply
    INSTALL(TARGETS ${name} DESTINATION /usr/lib)
ENDMACRO()

make_lib(modelSVM)



