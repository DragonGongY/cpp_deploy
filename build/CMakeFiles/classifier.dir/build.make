# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/dp/LinuxData/Algorithms/PaddleX-develop/deploy/cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/dp/LinuxData/Algorithms/PaddleX-develop/deploy/cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/classifier.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/classifier.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/classifier.dir/flags.make

CMakeFiles/classifier.dir/demo/classifier.cpp.o: CMakeFiles/classifier.dir/flags.make
CMakeFiles/classifier.dir/demo/classifier.cpp.o: ../demo/classifier.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/dp/LinuxData/Algorithms/PaddleX-develop/deploy/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/classifier.dir/demo/classifier.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/classifier.dir/demo/classifier.cpp.o -c /media/dp/LinuxData/Algorithms/PaddleX-develop/deploy/cpp/demo/classifier.cpp

CMakeFiles/classifier.dir/demo/classifier.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/classifier.dir/demo/classifier.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/dp/LinuxData/Algorithms/PaddleX-develop/deploy/cpp/demo/classifier.cpp > CMakeFiles/classifier.dir/demo/classifier.cpp.i

CMakeFiles/classifier.dir/demo/classifier.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/classifier.dir/demo/classifier.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/dp/LinuxData/Algorithms/PaddleX-develop/deploy/cpp/demo/classifier.cpp -o CMakeFiles/classifier.dir/demo/classifier.cpp.s

CMakeFiles/classifier.dir/demo/classifier.cpp.o.requires:

.PHONY : CMakeFiles/classifier.dir/demo/classifier.cpp.o.requires

CMakeFiles/classifier.dir/demo/classifier.cpp.o.provides: CMakeFiles/classifier.dir/demo/classifier.cpp.o.requires
	$(MAKE) -f CMakeFiles/classifier.dir/build.make CMakeFiles/classifier.dir/demo/classifier.cpp.o.provides.build
.PHONY : CMakeFiles/classifier.dir/demo/classifier.cpp.o.provides

CMakeFiles/classifier.dir/demo/classifier.cpp.o.provides.build: CMakeFiles/classifier.dir/demo/classifier.cpp.o


CMakeFiles/classifier.dir/src/transforms.cpp.o: CMakeFiles/classifier.dir/flags.make
CMakeFiles/classifier.dir/src/transforms.cpp.o: ../src/transforms.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/dp/LinuxData/Algorithms/PaddleX-develop/deploy/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/classifier.dir/src/transforms.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/classifier.dir/src/transforms.cpp.o -c /media/dp/LinuxData/Algorithms/PaddleX-develop/deploy/cpp/src/transforms.cpp

CMakeFiles/classifier.dir/src/transforms.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/classifier.dir/src/transforms.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/dp/LinuxData/Algorithms/PaddleX-develop/deploy/cpp/src/transforms.cpp > CMakeFiles/classifier.dir/src/transforms.cpp.i

CMakeFiles/classifier.dir/src/transforms.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/classifier.dir/src/transforms.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/dp/LinuxData/Algorithms/PaddleX-develop/deploy/cpp/src/transforms.cpp -o CMakeFiles/classifier.dir/src/transforms.cpp.s

CMakeFiles/classifier.dir/src/transforms.cpp.o.requires:

.PHONY : CMakeFiles/classifier.dir/src/transforms.cpp.o.requires

CMakeFiles/classifier.dir/src/transforms.cpp.o.provides: CMakeFiles/classifier.dir/src/transforms.cpp.o.requires
	$(MAKE) -f CMakeFiles/classifier.dir/build.make CMakeFiles/classifier.dir/src/transforms.cpp.o.provides.build
.PHONY : CMakeFiles/classifier.dir/src/transforms.cpp.o.provides

CMakeFiles/classifier.dir/src/transforms.cpp.o.provides.build: CMakeFiles/classifier.dir/src/transforms.cpp.o


CMakeFiles/classifier.dir/src/paddlex.cpp.o: CMakeFiles/classifier.dir/flags.make
CMakeFiles/classifier.dir/src/paddlex.cpp.o: ../src/paddlex.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/dp/LinuxData/Algorithms/PaddleX-develop/deploy/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/classifier.dir/src/paddlex.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/classifier.dir/src/paddlex.cpp.o -c /media/dp/LinuxData/Algorithms/PaddleX-develop/deploy/cpp/src/paddlex.cpp

CMakeFiles/classifier.dir/src/paddlex.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/classifier.dir/src/paddlex.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/dp/LinuxData/Algorithms/PaddleX-develop/deploy/cpp/src/paddlex.cpp > CMakeFiles/classifier.dir/src/paddlex.cpp.i

CMakeFiles/classifier.dir/src/paddlex.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/classifier.dir/src/paddlex.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/dp/LinuxData/Algorithms/PaddleX-develop/deploy/cpp/src/paddlex.cpp -o CMakeFiles/classifier.dir/src/paddlex.cpp.s

CMakeFiles/classifier.dir/src/paddlex.cpp.o.requires:

.PHONY : CMakeFiles/classifier.dir/src/paddlex.cpp.o.requires

CMakeFiles/classifier.dir/src/paddlex.cpp.o.provides: CMakeFiles/classifier.dir/src/paddlex.cpp.o.requires
	$(MAKE) -f CMakeFiles/classifier.dir/build.make CMakeFiles/classifier.dir/src/paddlex.cpp.o.provides.build
.PHONY : CMakeFiles/classifier.dir/src/paddlex.cpp.o.provides

CMakeFiles/classifier.dir/src/paddlex.cpp.o.provides.build: CMakeFiles/classifier.dir/src/paddlex.cpp.o


# Object files for target classifier
classifier_OBJECTS = \
"CMakeFiles/classifier.dir/demo/classifier.cpp.o" \
"CMakeFiles/classifier.dir/src/transforms.cpp.o" \
"CMakeFiles/classifier.dir/src/paddlex.cpp.o"

# External object files for target classifier
classifier_EXTERNAL_OBJECTS =

demo/classifier: CMakeFiles/classifier.dir/demo/classifier.cpp.o
demo/classifier: CMakeFiles/classifier.dir/src/transforms.cpp.o
demo/classifier: CMakeFiles/classifier.dir/src/paddlex.cpp.o
demo/classifier: CMakeFiles/classifier.dir/build.make
demo/classifier: /home/dp/envs/Paddle-develop/build/paddle_inference_install_dir/paddle/lib/libpaddle_fluid.so
demo/classifier: /home/dp/envs/Paddle-develop/build/paddle_inference_install_dir/third_party/install/mklml/lib/libmklml_intel.so
demo/classifier: /home/dp/envs/Paddle-develop/build/paddle_inference_install_dir/third_party/install/mklml/lib/libiomp5.so
demo/classifier: /home/dp/envs/Paddle-develop/build/paddle_inference_install_dir/third_party/install/mkldnn/lib/libmkldnn.so.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
demo/classifier: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
demo/classifier: CMakeFiles/classifier.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/dp/LinuxData/Algorithms/PaddleX-develop/deploy/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable demo/classifier"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/classifier.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/classifier.dir/build: demo/classifier

.PHONY : CMakeFiles/classifier.dir/build

CMakeFiles/classifier.dir/requires: CMakeFiles/classifier.dir/demo/classifier.cpp.o.requires
CMakeFiles/classifier.dir/requires: CMakeFiles/classifier.dir/src/transforms.cpp.o.requires
CMakeFiles/classifier.dir/requires: CMakeFiles/classifier.dir/src/paddlex.cpp.o.requires

.PHONY : CMakeFiles/classifier.dir/requires

CMakeFiles/classifier.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/classifier.dir/cmake_clean.cmake
.PHONY : CMakeFiles/classifier.dir/clean

CMakeFiles/classifier.dir/depend:
	cd /media/dp/LinuxData/Algorithms/PaddleX-develop/deploy/cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/dp/LinuxData/Algorithms/PaddleX-develop/deploy/cpp /media/dp/LinuxData/Algorithms/PaddleX-develop/deploy/cpp /media/dp/LinuxData/Algorithms/PaddleX-develop/deploy/cpp/build /media/dp/LinuxData/Algorithms/PaddleX-develop/deploy/cpp/build /media/dp/LinuxData/Algorithms/PaddleX-develop/deploy/cpp/build/CMakeFiles/classifier.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/classifier.dir/depend
