#ifndef vcl_algorithm_h_
#define vcl_algorithm_h_

#include "vcl_compiler.h"
#include <algorithm>
/* The following includes are needed to preserve backwards
   compatilibility for external applications.  Previously
   definitions were defined in multiple headers with conditional
   ifndef guards, but we now include a reference header
   instead */
//no dependancies remove comment above
//vcl alias names to std names
#define vcl_adjacent_find std::adjacent_find
#define vcl_and std::and
#define vcl_binary std::binary
#define vcl_binary_search std::binary_search
#define vcl_copy std::copy
#define vcl_copy_ std::copy_
#define vcl_count std::count
#define vcl_count_if std::count_if
#define vcl_equal std::equal
#define vcl_equal_range std::equal_range
#define vcl_fill std::fill
#define vcl_fill_n std::fill_n
#define vcl_find std::find
#define vcl_find_end std::find_end
#define vcl_find_first_of std::find_first_of
#define vcl_find_if std::find_if
#define vcl_for_each std::for_each
#define vcl_generate std::generate
#define vcl_generate_n std::generate_n
#define vcl_generators_ std::generators_
#define vcl_heap std::heap
#define vcl_includes std::includes
#define vcl_inplace_merge std::inplace_merge
#define vcl_iter_swap std::iter_swap
#define vcl_lexicographical_compare std::lexicographical_compare
#define vcl_lower_bound std::lower_bound
#define vcl_make_heap std::make_heap
#define vcl_max std::max
#define vcl_min std::min
#define vcl_max_element std::max_element
#define vcl_merge std::merge
#define vcl_merge_ std::merge_
#define vcl_min_element std::min_element
#define vcl_mismatch std::mismatch
#define vcl_next_permutation std::next_permutation
#define vcl_nth_element std::nth_element
#define vcl_partial_sort std::partial_sort
#define vcl_partial_sort_copy std::partial_sort_copy
#define vcl_partition std::partition
#define vcl_stable_partition std::stable_partition
#define vcl_partitions_ std::partitions_
#define vcl_pop_heap std::pop_heap
#define vcl_prev_permutation std::prev_permutation
#define vcl_push_heap std::push_heap
#define vcl_random_shuffle std::random_shuffle
#define vcl_remove std::remove
#define vcl_remove_copy std::remove_copy
#define vcl_remove_copy_if std::remove_copy_if
#define vcl_remove_if std::remove_if
#define vcl_replace std::replace
#define vcl_replace_copy std::replace_copy
#define vcl_replace_copy_if std::replace_copy_if
#define vcl_replace_if std::replace_if
#define vcl_reverse std::reverse
#define vcl_reverse_copy std::reverse_copy
#define vcl_rotate std::rotate
#define vcl_rotate_copy std::rotate_copy
#define vcl_search std::search
#define vcl_search_n std::search_n
#define vcl_set_difference std::set_difference
#define vcl_set_intersection std::set_intersection
#define vcl_set_symmetric_difference std::set_symmetric_difference
#define vcl_set_union std::set_union
#define vcl_sort std::sort
#define vcl_sort_ std::sort_
#define vcl_sort_heap std::sort_heap
#define vcl_stable_sort std::stable_sort
#define vcl_swap std::swap
#define vcl_swap_ std::swap_
#define vcl_swap_ranges std::swap_ranges
#define vcl_transform std::transform
#define vcl_unique std::unique
#define vcl_unique_copy std::unique_copy
#define vcl_upper_bound std::upper_bound

#endif // vcl_algorithm_h_
