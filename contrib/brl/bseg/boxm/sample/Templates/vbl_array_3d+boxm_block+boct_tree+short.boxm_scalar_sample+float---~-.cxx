#include <vcl_compiler.h>
#include <iostream>
#include <string>
#include <vbl/vbl_array_3d.hxx>
#include <vgl/io/vgl_io_point_3d.h>
#include <boxm/boxm_block.hxx>
#include <boxm/sample/boxm_scalar_sample.h>
#include <boct/boct_tree.h>

typedef boct_tree<short, boxm_scalar_sample<float> > tree_type;
VBL_ARRAY_3D_INSTANTIATE(boxm_block<tree_type>*);


