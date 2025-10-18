add_subdirectory(runtime)

set(JULIA_SRC_FILES
   )

foreach(file ${JULIA_SRC_FILES})
   if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
          install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/${file}"
                              DESTINATION "${CMAKE_INSTALL_PREFIX}/src")
   endif()
endforeach()

#foreach(file ${JULIA_SRC_FILES})
#   if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
#          install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/${file}"
#                              DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/iris")
#   endif()
#endforeach()


