# Compiler pedantic flags, as specified in Jed Brown's dohp code 

set(DEFAULT_PEDANTIC_FLAGS "-pedantic -Wall -Wextra -Wundef -Wshadow -Wpointer-arith -Wbad-function-cast -Wcast-align -Wwrite-strings -Wconversion -Wlogical-op -Wsign-compare -Wstrict-prototypes -Wmissing-prototypes -Wmissing-declarations -Wredundant-decls -Wnested-externs -Winline -Wno-long-long -Wmissing-format-attribute -Wmissing-noreturn -Wpacked -Wdisabled-optimization -Wmultichar -Wformat-nonliteral -Wformat-security -Wformat-y2k -Wendif-labels -Wdeclaration-after-statement -Wold-style-definition -Winvalid-pch -Wmissing-field-initializers -Wvariadic-macros -Wunsafe-loop-optimizations -Wvolatile-register-var -Wstrict-aliasing -funit-at-a-time -Wno-sign-conversion")
#set(DEFAULT_PEDANTIC_FLAGS "-Wunreachable-code -Wfloat-equal -Wc++-compat")
#set(DEFAULT_PEDANTIC_FLAGS "-pedantic -Wall -Wextra -Winline -Wshadow -Wconversion -Wlogical-op -Wmissing-prototypes -Wvla")
#set(DEFAULT_PEDANTIC_FLAGS "${DEFAULT_PEDANTIC_FLAGS} -Wno-sign-conversion -Wwrite-strings -Wstrict-aliasing -Wcast-align -fstrict-aliasing")
#set(DEFAULT_PEDANTIC_FLAGS "${DEFAULT_PEDANTIC_FLAGS} -Wdisabled-optimization -funit-at-a-time")
#set(DEFAULT_PEDANTIC_FLAGS "${DEFAULT_PEDANTIC_FLAGS} -Wpadded")
set(PROJECT_PEDANTIC_FLAGS ${DEFAULT_PEDANTIC_FLAGS} CACHE STRING "Compiler flags to enable pedantic warnings")

# adds flags to the compilation
if(${ENABLE_PEDANTIC})
  add_definitions(${PROJECT_PEDANTIC_FLAGS})
endif()