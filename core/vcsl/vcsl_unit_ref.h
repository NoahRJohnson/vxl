//*****************************************************************************
// File name: vcsl_unit_ref.h
// Description: Smart pointer on a vcsl_unit
//-----------------------------------------------------------------------------
// Language: C++
//
// Version |Date      | Author                   |Comment
// --------+----------+--------------------------+-----------------------------
// 1.0     |2000/06/28| Fran�ois BERTEL          |Creation
//*****************************************************************************
#ifndef VCSL_UNIT_REF_H
#define VCSL_UNIT_REF_H

class vcsl_unit;

//*****************************************************************************
// External declarations for values
//*****************************************************************************
#include <vbl/vbl_smart_ptr.h>

typedef vbl_smart_ptr<vcsl_unit> vcsl_unit_ref;

#endif // #ifndef VCSL_UNIT_REF_H
