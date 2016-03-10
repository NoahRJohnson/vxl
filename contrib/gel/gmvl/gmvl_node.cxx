// This is gel/gmvl/gmvl_node.cxx
#ifdef VCL_NEEDS_PRAGMA_INTERFACE
#pragma implementation
#endif
//:
// \file
// \author crossge@crd.ge.com

#include "gmvl_node.h"
#include <vcl_compiler.h>
#include <iostream>
#include <iostream>

std::ostream &operator<<( std::ostream &os, gmvl_node &node)
{
  return os << "<Node " << &node << " ref=" << node.ref_ << " type=\"" << node.type_ << "\">";
}
