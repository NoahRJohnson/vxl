#ifndef sdet_region_classifier_params_h_
#define sdet_region_classifier_params_h_
//:
// \file
// \brief parameter mixin for sdet_region_classifier
//
// \author
//    Joseph L. Mundy - Dec. 26, 2016
//    GE Corporate Research and Development
//
//-----------------------------------------------------------------------------
#include <iostream>
#include <iosfwd>
#include <gevd/gevd_param_mixin.h>
#include <sdet/sdet_detector_params.h>
#ifdef _MSC_VER
#  include <vcl_msvc_warnings.h>
#endif

class sdet_region_classifier_params : public gevd_param_mixin
{
 public:
 sdet_region_classifier_params():sigma_(1.0f), vd_noise_mul_(1.25f),
    four_or_eight_conn_(4), margin_(10), weight_thresh_(50), min_region_size_(10), nbins_(16),
    verbose_(true), debug_(false), use_vd_edges_(true){}

  sdet_region_classifier_params(bool use_vd_edges,
                               float sigma,
                               float vd_noise_mul,
                               int four_or_eight_conn,
                               unsigned margin,
                               double weight_thresh,
                               int min_region_size,
                               unsigned nbins,
                               bool verbose,
                               bool debug
                               );
  sdet_region_classifier_params(const sdet_region_classifier_params& old_params);
 ~sdet_region_classifier_params() override= default;

  bool SanityCheck() override;
  friend
    std::ostream& operator<<(std::ostream&, const sdet_region_classifier_params& rpp);
 protected:
  void InitParams(bool use_vd_edges,
                  float sigma,
                  float vd_noise_mul,
                  int four_or_eight_conn,
                  unsigned margin,
                  double weight_thresh,
                  int min_region_size,
                  unsigned nbins,
                  bool verbose,
                  bool debug);
 public:
  //
  // Parameter blocks and parameters
  //
  bool use_vd_edges_;
  float sigma_;
  float vd_noise_mul_;
  int four_or_eight_conn_;
  unsigned margin_;
  double weight_thresh_;
  int min_region_size_;
  unsigned nbins_;
  bool debug_;               //!< Carry out debug processing
  bool verbose_;             //!< Print detailed output
};

#endif // sdet_region_classifier_params_h_
