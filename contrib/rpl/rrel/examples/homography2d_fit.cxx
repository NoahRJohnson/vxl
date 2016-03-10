//:
// \file
// \author Chuck Stewart
// \author Bess Lee
//
// \brief Program demonstrating use of the Robust Estimation library in line fitting
//

#include <vcl_compiler.h>
#include <iostream>
#include <iostream>
#include <vector>
#include <vnl/vnl_vector.h>

#include <rrel/rrel_homography2d_est.h>
#include <rrel/rrel_lms_obj.h>
#include <rrel/rrel_lts_obj.h>
#include <rrel/rrel_ran_sam_search.h>
#include <rrel/rrel_ransac_obj.h>
#include <rrel/rrel_trunc_quad_obj.h>
#include <rrel/rrel_mlesac_obj.h>
#include <rrel/rrel_muset_obj.h>

int
main()
{
  //
  // Read in the data.
  //
  // homography_fit_20.dat contains data generated by homography_gen
  // with 20% gross outliers.
  //

  std::vector< vnl_vector<double> > from_pts;
  std::vector< vnl_vector<double> > to_pts;
  vnl_vector<double> p(3);
  double x1, x2, y1, y2, w1, w2;

  while (std::cin >> x1 >> y1 >> w1 >> x2 >> y2 >> w2 ) {
    p[0] = x1; p[1] = y1; p[2] = w1;
    from_pts.push_back(p);

    p[0] = x2; p[1] = y2; p[2] = w2;
    to_pts.push_back(p);
  }

  //
  // Construct the problem
  //

  rrel_homography2d_est * hg = new rrel_homography2d_est( from_pts, to_pts );

  double max_outlier_frac = 0.5;
  double desired_prob_good = 0.99;
  int max_pops = 1;
  int trace_level = 0;

  //
  //  Least Median of Squares
  //
  {
    int num_dep_res = hg->num_samples_to_instantiate();
    rrel_objective* lms = new rrel_lms_obj( num_dep_res );

    rrel_ran_sam_search* ransam = new rrel_ran_sam_search;
    ransam->set_trace_level(trace_level);
    ransam->set_sampling_params( max_outlier_frac, desired_prob_good, max_pops);

    if ( !ransam->estimate( hg, lms ) )
      std::cout << "LMS failed!!\n";
    else {
      std::cout << "LMS succeeded.\n"
               << "estimate = " << ransam->params() << std::endl
               << "scale = " << ransam->scale() << std::endl;
    }
    std::cout << std::endl;
    delete lms;
    delete ransam;
  }

  //
  //  RANSAC
  //
  {
    rrel_ransac_obj* ransac = new rrel_ransac_obj();
    hg->set_prior_scale( 1.0 );

    rrel_ran_sam_search* ransam = new rrel_ran_sam_search;
    ransam->set_trace_level(trace_level);
    ransam->set_sampling_params( max_outlier_frac, desired_prob_good, max_pops);

    if ( !ransam->estimate( hg, ransac ) )
      std::cout << "RANSAC failed!!\n";
    else {
      std::cout << "RANSAC succeeded.\n"
               << "estimate = " << ransam->params() << std::endl
               << "scale = " << ransam->scale() << std::endl;
    }
    std::cout << std::endl;
    delete ransac;
    delete ransam;
  }


  //
  //  MSAC
  //
  {
    rrel_trunc_quad_obj* msac = new rrel_trunc_quad_obj();
    rrel_ran_sam_search * ransam = new rrel_ran_sam_search;
    ransam->set_trace_level(trace_level);
    ransam->set_sampling_params( max_outlier_frac, desired_prob_good, max_pops);

    if ( !ransam->estimate( hg, msac ) )
      std::cout << "MSAC failed!!\n";
    else {
      std::cout << "MSAC succeeded.\n"
               << "estimate = " << ransam->params() << std::endl
               << "scale = " << ransam->scale() << std::endl;
    }
    std::cout << std::endl;
    delete msac;
    delete ransam;
  }

  //
  //  MLESAC
  //
  {
    int residual_dof = hg->residual_dof();
    rrel_mlesac_obj* mlesac = new rrel_mlesac_obj( residual_dof );

    hg->set_prior_scale(1.0);

    rrel_ran_sam_search * ransam = new rrel_ran_sam_search;
    ransam->set_trace_level(trace_level);
    ransam->set_sampling_params( max_outlier_frac, desired_prob_good, max_pops);

    if ( !ransam->estimate( hg, mlesac ) )
      std::cout << "MLESAC failed!!\n";
    else {
      std::cout << "MLESAC succeeded.\n"
               << "estimate = " << ransam->params() << std::endl
               << "scale = " << ransam->scale() << std::endl;
    }
    std::cout << std::endl;
    delete mlesac;
    delete ransam;
  }

  //
  //  MUSE
  //
  {
    hg->set_no_prior_scale();

    rrel_muset_obj* muset = new rrel_muset_obj( from_pts.size()+1 );
    rrel_ran_sam_search * ransam = new rrel_ran_sam_search;
    ransam->set_trace_level(trace_level);
    ransam->set_sampling_params( 1 - muset->min_inlier_fraction(),
                                 desired_prob_good,
                                 max_pops);

    if ( !ransam->estimate( hg, muset ) )
      std::cout << "MUSE failed!!\n";
    else {
      std::cout << "MUSE succeeded.\n"
               << "estimate = " << ransam->params() << std::endl
               << "scale = " << ransam->scale() << std::endl;
    }
    std::cout << std::endl;
    delete muset;
    delete ransam;
  }

  delete hg;

  return 0;
}
