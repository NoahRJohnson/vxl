// This is core/vil1/vil1_pyramid.cxx
#ifdef VCL_NEEDS_PRAGMA_INTERFACE
#pragma implementation
#endif
//:
// \file
// \author fsm

#include "vil1_pyramid.h"
#include <vcl_cassert.h>
#include <vcl_compiler.h>
#include <iostream>
#include <vil1/vil1_resample.h>
#include <vil1/vil1_memory_image.h>

vil1_pyramid::vil1_pyramid(vil1_image const &I, cache_strategy cs_)
{
  cs = cs_;
  levels.push_back(I);
}

vil1_pyramid::~vil1_pyramid()
{
}

vil1_image vil1_pyramid::operator[](unsigned i)
{
  while (i >= levels.size()) {
    std::cerr << "making pyramid level " << levels.size() << std::endl;
    vil1_image I = levels.back();
    I = vil1_resample(I, I.width()/2, I.height()/2);
    switch (cs) {
    case none: break;
    case memory: I = vil1_memory_image(I /*, "silly hack"*/); break;
    case blocked: /* not implemented yet */
    default: assert(!"not implemented"); break;
    }
    levels.push_back(I);
  }
  return levels[i];
}
