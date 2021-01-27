

set(command "${make}")
execute_process(
  COMMAND ${command}
  RESULT_VARIABLE result
  OUTPUT_FILE "/media/dp/LinuxData/Algorithms/PaddleX-develop/deploy/cpp/build/ext/yaml-cpp/src/ext-yaml-cpp-stamp/ext-yaml-cpp-build-out.log"
  ERROR_FILE "/media/dp/LinuxData/Algorithms/PaddleX-develop/deploy/cpp/build/ext/yaml-cpp/src/ext-yaml-cpp-stamp/ext-yaml-cpp-build-err.log"
  )
if(result)
  set(msg "Command failed: ${result}\n")
  foreach(arg IN LISTS command)
    set(msg "${msg} '${arg}'")
  endforeach()
  set(msg "${msg}\nSee also\n  /media/dp/LinuxData/Algorithms/PaddleX-develop/deploy/cpp/build/ext/yaml-cpp/src/ext-yaml-cpp-stamp/ext-yaml-cpp-build-*.log")
  message(FATAL_ERROR "${msg}")
else()
  set(msg "ext-yaml-cpp build command succeeded.  See also /media/dp/LinuxData/Algorithms/PaddleX-develop/deploy/cpp/build/ext/yaml-cpp/src/ext-yaml-cpp-stamp/ext-yaml-cpp-build-*.log")
  message(STATUS "${msg}")
endif()
