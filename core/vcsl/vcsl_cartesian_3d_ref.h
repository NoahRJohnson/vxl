//*****************************************************************************
// File name: vcsl_cartesian_3d_ref.h
// Description: Smart pointer on a vcsl_cartesian_3d
//-----------------------------------------------------------------------------
// Language: C++
//
// Version |Date      | Author                   |Comment
// --------+----------+--------------------------+-----------------------------
// 1.0     |2000/06/28| Fran�ois BERTEL          |Creation
//*****************************************************************************
#ifndef VCSL_CARTESIAN_3D_REF_H
#define VCSL_CARTESIAN_3D_REF_H

class vcsl_cartesian_3d;

//*****************************************************************************
// External declarations for values
//*****************************************************************************
#include <vbl/vbl_smart_ptr.h>

typedef vbl_smart_ptr<vcsl_cartesian_3d> vcsl_cartesian_3d_ref;

#endif // #ifndef VCSL_CARTESIAN_3D_REF_H
