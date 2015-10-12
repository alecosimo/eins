
if(BUILD_DOCUMENTATION)

  ## BUILDING MAN PAGES
  file(REMOVE_RECURSE ${CMAKE_DOC_OUTPUT_DIRECTORY})
  file(MAKE_DIRECTORY ${CMAKE_DOC_OUTPUT_DIRECTORY}/man/html)
  add_custom_command(
    OUTPUT    ${CMAKE_DOC_OUTPUT_DIRECTORY}/man/index.html
    COMMAND   ${PETSC_DIR}/${PETSC_ARCH}/bin/doctext 
    ARGS      -mpath ${CMAKE_DOC_OUTPUT_DIRECTORY}/man/html -html ${Eins_sources}
    COMMAND   echo '<TITLE>EINS Man Pages</TITLE>' > man/index.html
    COMMAND   echo '<H1>EINS Man Pages</H1>' >> man/index.html
    COMMAND   echo '<MENU>' >> man/index.html
    COMMAND   ls -1 man/html | sed -e "'s%^\\(.*\\).html$$%<LI><A HREF=\"html/\\1.html\">\\1</A>%g'" >> man/index.html
    COMMAND   echo '</MENU>' >> man/index.html
    COMMENT   "Building EINS man pages"
    WORKING_DIRECTORY ${CMAKE_DOC_OUTPUT_DIRECTORY}
    )
  add_custom_target(einsMAN ALL DEPENDS ${CMAKE_DOC_OUTPUT_DIRECTORY}/man/index.html)
 
  ## BUILDING LATEX DOCUMENTATION: have a look at teh CMake FAQs
  find_package(LATEX REQUIRED)

  if(LATEX_COMPILER)
    exec_program(kpsewhich ARGS minted.sty OUTPUT_VARIABLE output)

    if("${output}" STREQUAL "")
      message(FATAL_ERROR "ERROR: latex package minted.sty not found. It is needed for compiling the documentation.")
    endif()
    
    add_custom_command(
      OUTPUT    ${CMAKE_DOC_OUTPUT_DIRECTORY}/einsManual.pdf
      COMMAND   ${PDFLATEX_COMPILER}
      ARGS      -shell-escape -interaction=batchmode einsManual
      COMMAND   ${BIBTEX_COMPILER}
      ARGS      -terse einsManual    
      COMMAND   ${PDFLATEX_COMPILER}
      ARGS      -shell-escape -interaction=batchmode einsManual
      COMMAND   ${PDFLATEX_COMPILER}
      ARGS      -shell-escape -interaction=batchmode einsManual
      COMMAND   mv einsManual.pdf ${CMAKE_DOC_OUTPUT_DIRECTORY}/einsManual.pdf
      COMMAND   rm *.blg *.aux *.bbl *.log *.out *.toc
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/doc/latex
      COMMENT   "Building EINS Manual"
      )
    
    add_custom_target(einsManual ALL DEPENDS ${CMAKE_DOC_OUTPUT_DIRECTORY}/einsManual.pdf)
  endif(LATEX_COMPILER)    

endif(BUILD_DOCUMENTATION)
