#Bring the sources
file(GLOB SOURCES_MODEL_SVM ${PROJECT_SOURCE_DIR}/src/lib/modelSVM/*.cpp)

#Generate library
add_library(modelSVM SHARED ${SOURCES_MODEL_SVM})

#Add other libraries used to build the main library
target_link_libraries(modelSVM PRIVATE ${OpenCV_LIBS})

#Set an installation path, type "sudo make install" to apply
install(TARGETS modelSVM DESTINATION /usr/lib)