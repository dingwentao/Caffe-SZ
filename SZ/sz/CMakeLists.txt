aux_source_directory(src sz_sources)

add_library (SZ SHARED ${sz_sources})
target_link_libraries (SZ zlib zstd)
target_compile_options(SZ
	PRIVATE $<$<CONFIG:Debug>:-O3 -Wall -Wextra -Wpedantic -Wno-unused-parameter>
	)

install (TARGETS SZ LIBRARY DESTINATION lib ARCHIVE DESTINATION lib )
