/* Copyright (C) 2001-2021 Artifex Software, Inc.
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


/* RasterOp source device */
#include "gx.h"
#include "gserrors.h"
#include "gxdcolor.h"
#include "gxdevice.h"
#include "gdevmrop.h"

/* GC procedures */
private_st_device_rop_texture();
static ENUM_PTRS_BEGIN(device_rop_texture_enum_ptrs) {
    if (index < st_device_color_max_ptrs) {
        gs_ptr_type_t ptype =
            ENUM_SUPER_ELT(gx_device_rop_texture, st_device_color, texture, 0);

        if (ptype)
            return ptype;
        return ENUM_OBJ(NULL);	/* don't stop early */
    }
    ENUM_PREFIX(st_device_forward, st_device_color_max_ptrs);
} ENUM_PTRS_END
static RELOC_PTRS_BEGIN(device_rop_texture_reloc_ptrs) {
    RELOC_PREFIX(st_device_forward);
    RELOC_SUPER(gx_device_rop_texture, st_device_color, texture);
} RELOC_PTRS_END

/* Device for providing source data for RasterOp. */
static dev_proc_fill_rectangle(rop_texture_fill_rectangle);
static dev_proc_copy_mono(rop_texture_copy_mono);
static dev_proc_copy_color(rop_texture_copy_color);
static dev_proc_copy_planes(rop_texture_copy_planes);

/* The device descriptor. */
static void
rop_texture_initialize_device_procs(gx_device *dev)
{
    set_dev_proc(dev, get_initial_matrix,gx_forward_get_initial_matrix);
    set_dev_proc(dev, map_rgb_color, gx_forward_map_rgb_color);
    set_dev_proc(dev, map_color_rgb, gx_forward_map_color_rgb);
    set_dev_proc(dev, fill_rectangle, rop_texture_fill_rectangle);
    set_dev_proc(dev, copy_mono, rop_texture_copy_mono);
    set_dev_proc(dev, copy_color, rop_texture_copy_color);
    set_dev_proc(dev, get_params, gx_forward_get_params);
    set_dev_proc(dev, put_params, gx_forward_put_params);
    set_dev_proc(dev, map_cmyk_color, gx_forward_map_cmyk_color);
    set_dev_proc(dev, get_page_device, gx_forward_get_page_device);
    set_dev_proc(dev, copy_alpha, gx_no_copy_alpha);
    set_dev_proc(dev, get_clipping_box, gx_forward_get_clipping_box);
    set_dev_proc(dev, get_hardware_params, gx_forward_get_hardware_params);
    set_dev_proc(dev, get_color_mapping_procs, gx_forward_get_color_mapping_procs);
    set_dev_proc(dev, get_color_comp_index, gx_forward_get_color_comp_index);
    set_dev_proc(dev, encode_color, gx_forward_encode_color);
    set_dev_proc(dev, decode_color, gx_forward_decode_color);
    set_dev_proc(dev, fill_rectangle_hl_color, gx_forward_fill_rectangle_hl_color);
    set_dev_proc(dev, include_color_space, gx_forward_include_color_space);
    set_dev_proc(dev, fill_linear_color_scanline, gx_forward_fill_linear_color_scanline);
    set_dev_proc(dev, fill_linear_color_trapezoid, gx_forward_fill_linear_color_trapezoid);
    set_dev_proc(dev, fill_linear_color_triangle, gx_forward_fill_linear_color_triangle);
    set_dev_proc(dev, update_spot_equivalent_colors, gx_forward_update_spot_equivalent_colors);
    set_dev_proc(dev, ret_devn_params, gx_forward_ret_devn_params);
    set_dev_proc(dev, fillpage, gx_forward_fillpage);
    set_dev_proc(dev, dev_spec_op, gx_forward_dev_spec_op);
    set_dev_proc(dev, copy_planes, rop_texture_copy_planes);
    set_dev_proc(dev, get_profile, gx_forward_get_profile);
    set_dev_proc(dev, set_graphics_type_tag, gx_forward_set_graphics_type_tag);
}
static const gx_device_rop_texture gs_rop_texture_device = {
    std_device_std_body(gx_device_rop_texture,
                        rop_texture_initialize_device_procs,
                        "rop source",
                        0, 0, 1, 1),
    { 0 },
    0,				/* target */
    lop_default			/* log_op */
    /* */				/* texture */
};

/* Create a RasterOp source device. */
int
gx_alloc_rop_texture_device(gx_device_rop_texture ** prsdev, gs_memory_t * mem,
                            client_name_t cname)
{
    *prsdev = gs_alloc_struct(mem, gx_device_rop_texture,
                              &st_device_rop_texture, cname);
    return (*prsdev == 0 ? gs_note_error(gs_error_VMerror) : 0);
}

/* Initialize a RasterOp source device. */
/* 'target' device must not be NULL */
void
gx_make_rop_texture_device(gx_device_rop_texture * dev, gx_device * target,
             gs_logical_operation_t log_op, const gx_device_color * texture)
{
    /* Can never fail */
    (void)gx_device_init((gx_device *) dev,
                         (const gx_device *)&gs_rop_texture_device,
                         target->memory, true);
    gx_device_set_target((gx_device_forward *)dev, target);
    /* Drawing operations are defaulted, non-drawing are forwarded. */
    check_device_separable((gx_device *) dev);
    gx_device_fill_in_procs((gx_device *) dev);
    gx_device_copy_params((gx_device *)dev, target);
    dev->graphics_type_tag = target->graphics_type_tag;	/* Init from device */
    dev->log_op = log_op;
    dev->texture = *texture;
}

/* Fill a rectangle */
static int
rop_texture_fill_rectangle(gx_device * dev, int x, int y, int w, int h,
                           gx_color_index color)
{
    gx_device_rop_texture *const rtdev = (gx_device_rop_texture *)dev;
    gx_rop_source_t source;

    source.sdata = NULL;
    source.sourcex = 0;
    source.sraster = 0;
    source.id = gx_no_bitmap_id;
    source.scolors[0] = source.scolors[1] = color;
    source.planar_height = 0;
    source.use_scolors = true;
    return gx_device_color_fill_rectangle(&rtdev->texture,
                                          x, y, w, h, rtdev->target,
                                          rtdev->log_op, &source);
}

/* Copy a monochrome rectangle */
static int
rop_texture_copy_mono(gx_device * dev,
                const byte * data, int sourcex, int raster, gx_bitmap_id id,
                      int x, int y, int w, int h,
                      gx_color_index color0, gx_color_index color1)
{
    gx_device_rop_texture *const rtdev = (gx_device_rop_texture *)dev;
    gx_rop_source_t source;
    gs_logical_operation_t lop = rtdev->log_op;

    source.sdata = data;
    source.sourcex = sourcex;
    source.sraster = raster;
    source.id = id;
    source.scolors[0] = color0;
    source.scolors[1] = color1;
    source.planar_height = 0;
    source.use_scolors = true;
    /* Adjust the logical operation per transparent colors. */
    if (color0 == gx_no_color_index)
        lop = rop3_use_D_when_S_0(lop);
    else if (color1 == gx_no_color_index)
        lop = rop3_use_D_when_S_1(lop);
    return gx_device_color_fill_rectangle(&rtdev->texture,
                                          x, y, w, h, rtdev->target,
                                          lop, &source);
}

/* Copy a color rectangle */
static int
rop_texture_copy_color(gx_device * dev,
                const byte * data, int sourcex, int raster, gx_bitmap_id id,
                       int x, int y, int w, int h)
{
    gx_device_rop_texture *const rtdev = (gx_device_rop_texture *)dev;
    gx_rop_source_t source;

    source.sdata = data;
    source.sourcex = sourcex;
    source.sraster = raster;
    source.id = id;
    source.scolors[0] = source.scolors[1] = gx_no_color_index;
    source.planar_height = 0;
    source.use_scolors = false;
    return gx_device_color_fill_rectangle(&rtdev->texture,
                                          x, y, w, h, rtdev->target,
                                          rtdev->log_op, &source);
}

/* Copy a color rectangle */
static int
rop_texture_copy_planes(gx_device * dev,
                        const byte * data, int sourcex, int raster,
                        gx_bitmap_id id, int x, int y, int w, int h,
                        int plane_height)
{
    gx_device_rop_texture *const rtdev = (gx_device_rop_texture *)dev;
    gx_rop_source_t source;

    source.sdata = data;
    source.sourcex = sourcex;
    source.sraster = raster;
    source.id = id;
    source.scolors[0] = source.scolors[1] = gx_no_color_index;
    source.planar_height = plane_height;
    source.use_scolors = false;
    return gx_device_color_fill_rectangle(&rtdev->texture,
                                          x, y, w, h, rtdev->target,
                                          rtdev->log_op,
                                          &source);
}
