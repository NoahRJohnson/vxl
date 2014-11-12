// This is brl/bseg/boxm2/cpp/pro/processes/boxm2_cpp_render_depth_of_max_prob_process.cxx
#include <bprb/bprb_func_process.h>
//:
// \file
// \brief  A process for rendering the depth of the surface which has the max prob of being the first visible and occupied surface along the ray
//
// \author Ozge C. Ozcanli
// \date Oct 23, 2014

#include <vcl_fstream.h>
#include <boxm2/io/boxm2_cache.h>
#include <boxm2/boxm2_scene.h>
#include <boxm2/boxm2_block.h>
#include <boxm2/boxm2_data_base.h>
#include <vil/vil_image_view.h>
#include <vil/vil_math.h>
//brdb stuff
#include <brdb/brdb_value.h>
#include <boxm2/cpp/algo/boxm2_render_functions.h>

//directory utility
#include <vcl_where_root_dir.h>

namespace boxm2_cpp_render_depth_of_max_prob_process_globals
{
  const unsigned n_inputs_ = 5;
  const unsigned n_outputs_ = 3;
  vcl_size_t lthreads[2]={8,8};
}

bool boxm2_cpp_render_depth_of_max_prob_process_cons(bprb_func_process& pro)
{
  using namespace boxm2_cpp_render_depth_of_max_prob_process_globals;

  //process takes 1 input
  vcl_vector<vcl_string> input_types_(n_inputs_);
  input_types_[0] = "boxm2_scene_sptr";
  input_types_[1] = "boxm2_cache_sptr";
  input_types_[2] = "vpgl_camera_double_sptr";
  input_types_[3] = "unsigned";
  input_types_[4] = "unsigned";


  // process has 1 output:
  // output[0]: scene sptr
  vcl_vector<vcl_string>  output_types_(n_outputs_);
  output_types_[0] = "vil_image_view_base_sptr";
  output_types_[1] = "vil_image_view_base_sptr";
  output_types_[2] = "vil_image_view_base_sptr";

  return pro.set_input_types(input_types_) && pro.set_output_types(output_types_);
}

bool boxm2_cpp_render_depth_of_max_prob_process(bprb_func_process& pro)
{
  using namespace boxm2_cpp_render_depth_of_max_prob_process_globals;
  
  if ( pro.n_inputs() < n_inputs_ ) {
    vcl_cout << pro.name() << ": The input number should be " << n_inputs_<< vcl_endl;
    return false;
  }
  //get the inputs
  unsigned i = 0;
  boxm2_scene_sptr scene =pro.get_input<boxm2_scene_sptr>(i++);
  boxm2_cache_sptr cache= pro.get_input<boxm2_cache_sptr>(i++);
  vpgl_camera_double_sptr cam= pro.get_input<vpgl_camera_double_sptr>(i++);
  unsigned ni=pro.get_input<unsigned>(i++);
  unsigned nj=pro.get_input<unsigned>(i++);

  // function call
  vil_image_view<float> * exp_img=new vil_image_view<float>(ni,nj);
  vil_image_view<float> * vis_img=new vil_image_view<float>(ni,nj);
  vil_image_view<float> * prob_img=new vil_image_view<float>(ni,nj);
  exp_img->fill(scene->bounding_box().min_z());
  vis_img->fill(1.0f);
  prob_img->fill(0.0f);
  vcl_vector<boxm2_block_id> vis_order=scene->get_vis_blocks((vpgl_generic_camera<double>*)(cam.ptr()));
  vcl_vector<boxm2_block_id>::iterator id;
  for (id = vis_order.begin(); id != vis_order.end(); ++id)
  {
    vcl_cout<<"Block Id "<<(*id)<<vcl_endl;
    boxm2_block *     blk  =  cache->get_block(*id);
    boxm2_data_base *  alph = cache->get_data_base(*id,boxm2_data_traits<BOXM2_ALPHA>::prefix());

    boxm2_scene_info_wrapper *scene_info_wrapper=new boxm2_scene_info_wrapper();
    scene_info_wrapper->info=scene->get_blk_metadata(*id);

    boxm2_render_depth_of_max_prob(scene_info_wrapper->info,
                                blk,alph,cam,exp_img,vis_img,prob_img,ni,nj);
  }

  // store scene smaprt pointer
  pro.set_output_val<vil_image_view_base_sptr>(0, exp_img);
  pro.set_output_val<vil_image_view_base_sptr>(1, prob_img);
  pro.set_output_val<vil_image_view_base_sptr>(2, vis_img);
  
  return true;
}
