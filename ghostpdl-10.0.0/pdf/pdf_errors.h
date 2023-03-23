/* Copyright (C) 2022 Artifex Software, Inc.
   All Rights Reserved.

   This software is provided AS-IS with no warranty, either express or
   implied.

   This software is distributed under license and may not be copied,
   modified or distributed except as expressly authorized under the terms
   of the license contained in the file LICENSE in this distribution.

   Refer to licensing information at http://www.artifex.com or contact
   Artifex Software, Inc.,  1305 Grant Avenue - Suite 200, Novato,
   CA 94945, U.S.A., +1(415)492-9861, for further information.
*/

#ifndef PARAM
#define PARAM(A,B) A
#endif
PARAM(E_PDF_NOERROR,                   "no error"),
PARAM(E_PDF_NOHEADER,                  "no header detected"),
PARAM(E_PDF_NOHEADERVERSION,           "header lacks a version number"),
PARAM(E_PDF_NOSTARTXREF,               "no startxref token found"),
PARAM(E_PDF_BADSTARTXREF,              "startxref offset invalid"),
PARAM(E_PDF_BADXREFSTREAM,             "couldn't read hybrid file's XrefStm"),
PARAM(E_PDF_BADXREF,                   "error in xref table"),
PARAM(E_PDF_SHORTXREF,                 "too few entries in xref table"),
PARAM(E_PDF_PREV_NOT_XREF_STREAM,      "The /Prev entry in an XrefStm dictionary did not point to an XrefStm"),
PARAM(E_PDF_MISSINGENDSTREAM,          "content stream lacks endstream"),
PARAM(E_PDF_UNKNOWNFILTER,             "request for unknown filter"),
PARAM(E_PDF_MISSINGWHITESPACE,         "missing white space after number"),
PARAM(E_PDF_MALFORMEDNUMBER,           "malformed number"),
PARAM(E_PDF_UNESCAPEDSTRING,           "unbalanced or unescaped character '(' in string"),
PARAM(E_PDF_BADOBJNUMBER,              "invalid object number"),
PARAM(E_PDF_MISSINGENDOBJ,             "object lacks an endobj"),
PARAM(E_PDF_TOKENERROR,                "error executing PDF token"),
PARAM(E_PDF_KEYWORDTOOLONG,            "potential token is too long"),
PARAM(E_PDF_BADPAGETYPE,               "Page object doe snot have /Page type"),
PARAM(E_PDF_CIRCULARREF,               "circular reference to indirect object"),
PARAM(E_PDF_UNREPAIRABLE,              "couldn't repair PDF file"),
PARAM(E_PDF_REPAIRED,                  "PDF file was repaired"),
PARAM(E_PDF_BADSTREAM,                 "error reading a stream"),
PARAM(E_PDF_MISSINGOBJ,                "obj token missing"),
PARAM(E_PDF_BADPAGEDICT,               "error in Page dictionary"),
PARAM(E_PDF_OUTOFMEMORY,               "out of memory"),
PARAM(E_PDF_PAGEDICTERROR,             "error reading page dictionary"),
PARAM(E_PDF_STACKUNDERFLOWERROR,       "stack underflow"),
PARAM(E_PDF_BADSTREAMDICT,             "error in stream dictionary"),
PARAM(E_PDF_INHERITED_STREAM_RESOURCE, "stream inherited a resource"),
PARAM(E_PDF_DEREF_FREE_OBJ,            "counting down reference to freed object"),
PARAM(E_PDF_INVALID_TRANS_XOBJECT,     "error in transparency XObject"),
PARAM(E_PDF_NO_SUBTYPE,                "object lacks a required Subtype"),
PARAM(E_PDF_IMAGECOLOR_ERROR,          "error in image colour"),
PARAM(E_DICT_SELF_REFERENCE,           "dictionary contains a key which (indirectly) references the dictionary."),
PARAM(E_IMAGE_MASKWITHCOLOR,           "Image has both ImageMask and ColorSpace keys."),
PARAM(E_PDF_INVALID_DECRYPT_LEN,       "Invalid /Length in Encryption dictionary (not in range 40-128 or not a multiple of 8)."),
PARAM(E_PDF_GROUP_NO_CS,               "Group attributes dictionary is missing /CS"),
PARAM(E_BAD_GROUP_DICT,                "Error retrieving Group dictionary for a page or XObject"),
PARAM(E_BAD_HALFTONE,                  "Error setting a halftone"),
PARAM(E_PDF_BADENCRYPT,                "Encrypt diciotnary not a dictionary"),
PARAM(E_PDF_MISSINGTYPE,               "A dictionary is missing a required /Type key."),
PARAM(E_PDF_NESTEDTOODEEP,             "Dictionaries/arrays nested too deeply"),
PARAM(E_PDF_BADPAGECOUNT,              "page tree root node /Count did not match the actual number of pages in the tree."),
#undef PARAM
