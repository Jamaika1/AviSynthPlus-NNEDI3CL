#include <cerrno>
#include <cstdio>

#include <locale>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#define BOOST_COMPUTE_HAVE_THREAD_LOCAL
#define BOOST_COMPUTE_THREAD_SAFE
#define BOOST_COMPUTE_USE_OFFLINE_CACHE
#include "boost/compute/core.hpp"
#include "boost/compute/utility/dim.hpp"
#include "boost/dll.hpp"

#include "../../avisynth.h"
#include "NNEDI3CL.cl"

struct NNEDI3CLData
{
    //AVS_FilterInfo* fi;
    boost::compute::command_queue queue;
    boost::compute::kernel kernel;
    boost::compute::image2d src;
    boost::compute::image2d dst;
    boost::compute::image2d tmp;
    boost::compute::buffer weights0;
    boost::compute::buffer weights1Buffer;
    cl_mem weights1;
    std::string err;
    int field;
    bool dh;
    bool dw;
    int process[4];
};

class Nnedi3CL : public GenericVideoFilter
{
    //std::string cplace;
    //Lut* init_lut;
    //std::vector<EWAPixelCoeff_Nnedi3CL*> out1;
    //int planecount;
    //float peak;
    int process[4];
    boost::compute::command_queue queue;
    boost::compute::kernel kernel;
    boost::compute::image2d src;
    boost::compute::image2d dst;
    boost::compute::image2d tmp;
    boost::compute::buffer weights0;
    boost::compute::buffer weights1Buffer;
    cl_mem weights1;
    std::string err;

    template<typename T, bool st>
    void resize_plane_c(PVideoFrame& src, PVideoFrame& dst, const int field_n, const NNEDI3CLData* const __restrict__ d);

    void(Nnedi3CL::*filter)(PVideoFrame&, PVideoFrame&, const int, const NNEDI3CLData* const __restrict__);

public:
    Nnedi3CL(PClip _child, int field, bool dh, bool dw, int planes1, int nsize, int nns, int qual, int etype, int pscrn, int device,
    bool list_device, bool info, bool st, bool luma, IScriptEnvironment* env);
    //PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
    int __stdcall SetCacheHints(int cachehints, int frame_range)
    {
        return cachehints == CACHE_GET_MTMODE ? MT_MULTI_INSTANCE : 0;
    }
    //~Nnedi3CL();
};

//#include "NNEDI3CL.h"

static constexpr int numNSIZE{ 7 };
static constexpr int numNNS{ 5 };
static constexpr int xdiaTable[numNSIZE]{ 8, 16, 32, 48, 8, 16, 32 };
static constexpr int ydiaTable[numNSIZE]{ 6, 6, 6, 6, 4, 4, 4 };
static constexpr int nnsTable[numNNS]{ 16, 32, 64, 128, 256 };

static std::mutex mtx;

static AVS_FORCEINLINE int roundds(const double f) noexcept
{
    return (f - std::floor(f) >= 0.5) ? std::min(static_cast<int>(std::ceil(f)), 32767) : std::max(static_cast<int>(std::floor(f)), -32768);
}

template<typename T, bool st>
void Nnedi3CL::resize_plane_c(PVideoFrame& src, PVideoFrame& dst, const int field_n, const NNEDI3CLData* const __restrict__ d)
{
    constexpr int planes_y[4]{ PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    constexpr int planes_r[4]{ PLANAR_R, PLANAR_G, PLANAR_B, PLANAR_A };
    const int* planes = (vi.IsRGB()) ? planes_r : planes_y;

    int planecount = vi.NumComponents();
    for (int i = 0; i < planecount; ++i)
    {
        const int plane = planes[i];
        if (d->process[i])
        {
            const int src_width = src->GetRowSize(plane) / sizeof(T);
            const int dst_width = dst->GetRowSize(plane) / sizeof(T);
            const int src_height = src->GetHeight(plane);
            const int dst_height = dst->GetHeight(plane);
            const T* srcp = reinterpret_cast<const T*>(src->GetReadPtr(plane));
            T* __restrict__ dstp = reinterpret_cast<T* __restrict__>(dst->GetWritePtr(plane));

            auto queue{ d->queue };
            auto kernel{ d->kernel };
            auto src_image{ d->src };
            auto dst_image{ d->dst };
            auto tmp_image{ d->tmp };

            constexpr size_t localWorkSize[2]{ 4, 16 };

            queue.enqueue_write_image(src_image, boost::compute::dim(0, 0), boost::compute::dim(src_width, src_height), srcp, src->GetPitch(planes[i]));

            if (d->dh && d->dw)
            {
                size_t globalWorkSize[]{ static_cast<size_t>(((src_height + 7) / 8 + 3) & -4), static_cast<size_t>((dst_width / 2 + 15) & -16) };
                kernel.set_args(src_image, tmp_image, d->weights0, d->weights1, src_height, src_width, src_height, dst_width, field_n, 1 - field_n, -1);
                queue.enqueue_nd_range_kernel(kernel, 2, nullptr, globalWorkSize, localWorkSize);

                globalWorkSize[0] = static_cast<size_t>(((dst_width + 7) / 8 + 3) & -4);
                globalWorkSize[1] = static_cast<size_t>((dst_height / 2 + 15) & -16);
                kernel.set_args(tmp_image, dst_image, d->weights0, d->weights1, dst_width, src_height, dst_width, dst_height, field_n, 1 - field_n, 0);
                queue.enqueue_nd_range_kernel(kernel, 2, nullptr, globalWorkSize, localWorkSize);
            }
            else if (d->dw)
            {
                const size_t globalWorkSize[]{ static_cast<size_t>(((dst_height + 7) / 8 + 3) & -4), static_cast<size_t>((dst_width / 2 + 15) & -16) };
                kernel.set_args(src_image, dst_image, d->weights0, d->weights1, src_height, src_width, dst_height, dst_width, field_n, 1 - field_n, -1);
                queue.enqueue_nd_range_kernel(kernel, 2, nullptr, globalWorkSize, localWorkSize);
            }
            else
            {
                const size_t globalWorkSize[]{ static_cast<size_t>(((dst_width + 7) / 8 + 3) & -4), static_cast<size_t>((dst_height / 2 + 15) & -16) };
                kernel.set_args(src_image, dst_image, d->weights0, d->weights1, src_width, src_height, dst_width, dst_height, field_n, 1 - field_n, 0);
                queue.enqueue_nd_range_kernel(kernel, 2, nullptr, globalWorkSize, localWorkSize);
            }

            if constexpr (st)
            {
                std::lock_guard<std::mutex> lck(mtx);
                queue.enqueue_read_image(dst_image, boost::compute::dim(0, 0), boost::compute::dim(dst_width, dst_height), dstp, dst->GetPitch(planes[i]));
            }
            else
                queue.enqueue_read_image(dst_image, boost::compute::dim(0, 0), boost::compute::dim(dst_width, dst_height), dstp, dst->GetPitch(planes[i]));
        }
    }
}

/* multiplies and divides a rational number, such as a frame duration, in place and reduces the result */
AVS_FORCEINLINE void muldivRational(int64_t* num, int64_t* den, int mul, int div)
{
    /* do nothing if the rational number is invalid */
    if (!*den)
        return;

    int64_t a;
    int64_t b;
    *num *= mul;
    *den *= div;
    a = *num;
    b = *den;

    while (b != 0)
    {
        int64_t t{ a };
        a = b;
        b = t % b;
    }

    if (a < 0)
        a = -a;

    *num /= a;
    *den /= a;
}

/*PVideoFrame* AVSC_CC NNEDI3CL_get_frame(AVS_FilterInfo* fi, int n)
{
    NNEDI3CLData* d{ static_cast<NNEDI3CLData*>(fi->user_data) };

    const int field_no_prop = [&]()
    {
        if (d->field == -1)
            return avs_get_parity(fi->child, n) ? 1 : 0;
        else if (d->field == -2)
            return avs_get_parity(fi->child, n >> 1) ? 3 : 2;
        else
            return -1;
    }();

    int field{ (d->field > -1) ? d->field : field_no_prop };

    PVideoFrame* src = child->GetFrame((field > 1) ? (n >> 1) : n, env);

    if (!src)
        return nullptr;

    PVideoFrame* dst = avs_new_video_frame_p(fi->env, &fi->vi, src);

    if (d->field < 0)
    {
        int err;
        const int64_t field_based{ env->propGetInt("_FieldBased", 0, &err) };
        if (err == 0)
        {
            if (field_based == 1)
                field = 0;
            else if (field_based == 2)
                field = 1;

            if (d->field > 1 || field_no_prop > 1)
            {
                if (field_based == 0)
                    field -= 2;

                field = static_cast<int>((n & 1) ? (field == 0) : (field == 1));
            }
        }
        else
        {
            if (field > 1)
            {
                field -= 2;
                field = static_cast<int>((n & 1) ? (field == 0) : (field == 1));
            }
        }
    }
    else
    {
        if (field > 1)
        {
            field -= 2;
            field = static_cast<int>((n & 1) ? (field == 0) : (field == 1));
        }
    }

    try
    {
        d->filter(src, dst, field, d);
    }
    catch (const boost::compute::opencl_error& error)
    {
        d->err = "NNEDI3CL: " + error.error_string();
        fi->error = d->err.c_str();
        avs_release_video_frame(src);
        avs_release_video_frame(dst);

        return nullptr;
    }

    AVS_Map* props{ env->getFramePropsRW(dst) };

    env->propGetInt(props, "_FieldBased", 0, 0);

    if (d->field > 1 || field_no_prop > 1)
    {
        int errNum;
        int errDen;
        int64_t durationNum{ env->propGetInt(props, "_DurationNum", 0, &errNum) };
        int64_t durationDen{ env->propGetInt(props, "_DurationDen", 0, &errDen) };
        if (errNum == 0 && errDen == 0)
        {
            muldivRational(&durationNum, &durationDen, 1, 2);
            env->propGetInt("_DurationNum", durationNum, 0);
            env->propGetInt("_DurationDen", durationDen, 0);
        }
    }

    //avs_release_video_frame(src);

    return dst;
}*/

/*void AVSC_CC free_NNEDI3CL(AVS_FilterInfo* fi)
{
    NNEDI3CLData* d{ static_cast<NNEDI3CLData*>(fi->user_data) };
    clReleaseMemObject(d->weights1);
    delete d;
}

int AVSC_CC NNEDI3CL_set_cache_hints(AVS_FilterInfo* fi, int cachehints, int frame_range)
{
    return cachehints == AVS_CACHE_GET_MTMODE ? 2 : 0;
}*/

//AVS_Value AVSC_CC Create_NNEDI3CL(AVS_ScriptEnvironment* env, AVS_Value args, void* param)
Nnedi3CL::Nnedi3CL(PClip _child, int field, bool dh, bool dw, int planes1, int nsize, int nns, int qual, int etype, int pscrn, int device1,
    bool list_device, bool info, bool st, bool luma, IScriptEnvironment* env)
    : GenericVideoFilter(_child)//, cplace(cplace_)
{
    //enum { Clip, Field, Dh, Dw, Planes, Nsize, Nns, Qual, Etype, Pscrn, Device, List_device, Info, St, Luma };

    //NNEDI3CLData* params{ new NNEDI3CLData() };

    //AVS_Clip* clip{ avs_new_c_filter(env, &fi, avs_array_elt(args, Clip), 1) };
    //const AVS_VideoInfo& vi_temp{ fi->vi };
    //AVS_Value v{ avs_void };

    int planecount = (planes1 == 0) ? vi.NumComponents() : planes1;

    NNEDI3CLData params =
    {
      queue,
      kernel,
      src,
      dst,
      tmp,
      weights0,
      weights1Buffer,
      weights1,
      err,
      field,
      dh,
      dw,
      process[4],
    };

    //int n = 0;
    params.field = field;
    params.dh = dh;
    params.dw = dw;
    //PVideoFrame prop1 = child->GetFrame((field > 1) ? (n >> 1) : n, env);
    //const AVSMap* props = env->getFramePropsRO(prop1);

    //if (!prop1)
        //env->ThrowError("field = 0");

    //PVideoFrame prop2 = env->NewVideoFrameP(vi, &prop1);

    /*if (params.field < 0)
    {
        int err;
        virtual int64_t __stdcall field_based = env->propGetInt("_FieldBased", 0, &err);
        if (err == 0)
        {
            if (field_based == 1)
                field = 0;
            else if (field_based == 2)
                field = 1;

            if (params.field > 1 || env->propGetType(props, "_ChromaLocation") =! "i")
            {
                if (field_based == 0)
                    field -= 2;

                field = static_cast<int>((n & 1) ? (field == 0) : (field == 1));
            }
        }
        else
        {
            if (field > 1)
            {
                field -= 2;
                field = static_cast<int>((n & 1) ? (field == 0) : (field == 1));
            }
        }
    }
    else
    {
        if (field > 1)
        {
            field -= 2;
            field = static_cast<int>((n & 1) ? (field == 0) : (field == 1));
        }
    }

    try
    {
        this.*filter(&prop1, &prop2, field, params);
    }
    catch (const boost::compute::opencl_error& error)
    {
        params.err = "NNEDI3CL: " + error.error_string();
        //fi->error = d->err.c_str();
        //avs_release_video_frame(src);
        //avs_release_video_frame(dst);
        env->ThrowError(params.err.c_str());
    }

    props = env->getFramePropsRW(prop2) };

    env->propGetInt(props, "_FieldBased", 0, 0);

    if (params.field > 1 || env->propGetInt(props, "_FieldBased", 0, 0) > 1)
    {
        int errNum;
        int errDen;
        int64_t durationNum = env->propGetInt(props, "_DurationNum", 0, &errNum);
        int64_t durationDen = env->propGetInt(props, "_DurationDen", 0, &errDen);
        if (errNum == 0 && errDen == 0)
        {
            muldivRational(&durationNum, &durationDen, 1, 2);
            env->propGetInt("_DurationNum", durationNum, 0);
            env->propGetInt("_DurationDen", durationDen, 0);
        }
    }*/

    try
    {
        /*if (!avs_check_version(env, 9))
        {
            if (avs_check_version(env, 10))
            {
                if (avs_get_env_property(env, AVS_AEP_INTERFACE_BUGFIX) < 2)
                    throw "AviSynth + version must be r3688 or later.";
            }
        }
        else
            throw "AviSynth+ version must be r3688 or later.";*/

        if (!vi.IsPlanar())
            env->ThrowError("only planar format is supported");

        //const int planes{ (avs_defined(avs_array_elt(args, Planes))) ? avs_array_size(avs_array_elt(args, Planes)) : 0 };
        constexpr int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
        constexpr int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
       	const int* planes = (vi.IsRGB()) ? planes_r : planes_y;

        for (int i = 0; i < 4; ++i)
            params.process[i] = (planecount <= 0);

        for (int i = 0; i < planecount; ++i)
        {
            //const int n{ avs_as_int(*(avs_as_array(avs_array_elt(args, Planes)) + i)) };
            const int n = planes[i];

            if (n >= vi.NumComponents())
                throw "plane index out of range";

            if (params.process[n])
                throw "plane specified twice";

            params.process[n] = 1;
        }

        //const int onlyY{ (avs_defined(avs_array_elt(args, Luma))) ? avs_as_bool(avs_array_elt(args, Luma)) : 0 };

        if (vi.IsY() && !vi.IsRGB())
        {
            if (planecount > 1)
                throw "luma cannot be true when processed planes are more than 1";
            if (!params.process[0])
                throw "planes=0 must be used for luma=true";
            vi.pixel_type |= VideoInfo::CS_GENERIC_Y;
        }

        //const int nsize{ avs_defined(avs_array_elt(args, Nsize)) ? avs_as_int(avs_array_elt(args, Nsize)) : 6 };
        //const int nns{ avs_defined(avs_array_elt(args, Nns)) ? avs_as_int(avs_array_elt(args, Nns)) : 1 };
        //const int qual{ avs_defined(avs_array_elt(args, Qual)) ? avs_as_int(avs_array_elt(args, Qual)) : 1 };
        //const int etype{ avs_defined(avs_array_elt(args, Etype)) ? avs_as_int(avs_array_elt(args, Etype)) : 0 };
        //const int pscrn{ avs_defined(avs_array_elt(args, Pscrn)) ? avs_as_int(avs_array_elt(args, Pscrn)) : (avs_component_size(&fi->vi) < 4) ? 2 : 1 };
        //const int device_id{ avs_defined(avs_array_elt(args, Device)) ? avs_as_int(avs_array_elt(args, Device)) : -1 };

        if (params.field < -2 || params.field > 3)
            throw "field must be -2, -1, 0, 1, 2 or 3";
        if (!params.dh && (vi.height & 1))
            throw "height must be mod 2 when dh=False";
        if (params.dh && params.field > 1)
            throw "field must be 0 or 1 when dh=True";
        if (params.dw && params.field > 1)
            throw "field must be 0 or 1 when dw=True";
        if (nsize < 0 || nsize > 6)
            throw "nsize must be 0, 1, 2, 3, 4, 5 or 6";
        if (nns < 0 || nns > 4)
            throw "nns must be 0, 1, 2, 3 or 4";
        if (qual < 1 || qual > 2)
            throw "qual must be 1 or 2";
        if (etype < 0 || etype > 1)
            throw "etype must be 0 or 1";

        if (vi.NumComponents() < 4)
        {
            if (pscrn < 1 || pscrn > 2)
                throw "pscrn must be 1 or 2";
        }
        else
        {
            if (pscrn != 1)
                throw "pscrn must be 1 for float input";
        }

        if (device1 >= static_cast<int>(boost::compute::system::device_count()))
            throw "device index out of range";

        if (list_device)
        {
            const auto devices{ boost::compute::system::devices() };

            for (size_t i{ 0 }; i < devices.size(); ++i)
                params.err += std::to_string(i) + ": " + devices[i].name() + " (" + devices[i].platform().name() + ")" + "\n";

            //AVS_Value cl{ avs_new_value_clip(clip) };
            //AVS_Value args_[2]{ cl , avs_new_value_string(err.c_str()) };
            //v = env->Invoke("Text", avs_new_value_array(args_, 2), 0);

            //avs_release_value(cl);
            //avs_release_clip(clip);

            //return v;
        }

        boost::compute::device device{ boost::compute::system::default_device() };

        if (device1 > -1)
            device = boost::compute::system::devices().at(device1);

        boost::compute::context context{ device };
        params.queue = boost::compute::command_queue{ context, device };

        if (info)
        {
            params.err = "=== Platform Info ===\n";
            const auto platform{ device.platform() };
            params.err += "Profile: " + platform.get_info<CL_PLATFORM_PROFILE>() + "\n";
            params.err += "Version: " + platform.get_info<CL_PLATFORM_VERSION>() + "\n";
            params.err += "Name: " + platform.get_info<CL_PLATFORM_NAME>() + "\n";
            params.err += "Vendor: " + platform.get_info<CL_PLATFORM_VENDOR>() + "\n";

            params.err += "\n";

            params.err += "=== Device Info ===\n";
            params.err += "Name: " + device.get_info<CL_DEVICE_NAME>() + "\n";
            params.err += "Vendor: " + device.get_info<CL_DEVICE_VENDOR>() + "\n";
            params.err += "Profile: " + device.get_info<CL_DEVICE_PROFILE>() + "\n";
            params.err += "Version: " + device.get_info<CL_DEVICE_VERSION>() + "\n";
            params.err += "Max compute units: " + std::to_string(device.get_info<CL_DEVICE_MAX_COMPUTE_UNITS>()) + "\n";
            params.err += "Max work-group size: " + std::to_string(device.get_info<CL_DEVICE_MAX_WORK_GROUP_SIZE>()) + "\n";
            const auto max_work_item_sizes{ device.get_info<CL_DEVICE_MAX_WORK_ITEM_SIZES>() };
            params.err += "Max work-item sizes: " + std::to_string(max_work_item_sizes[0]) + ", " + std::to_string(max_work_item_sizes[1]) + ", " + std::to_string(max_work_item_sizes[2]) + "\n";
            params.err += "2D image max width: " + std::to_string(device.get_info<CL_DEVICE_IMAGE2D_MAX_WIDTH>()) + "\n";
            params.err += "2D image max height: " + std::to_string(device.get_info<CL_DEVICE_IMAGE2D_MAX_HEIGHT>()) + "\n";
            params.err += "Image support: " + std::string{ device.get_info<CL_DEVICE_IMAGE_SUPPORT>() ? "CL_TRUE" : "CL_FALSE" } + "\n";
            const auto global_mem_cache_type{ device.get_info<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE>() };
            if (global_mem_cache_type == CL_NONE)
                params.err += "Global memory cache type: CL_NONE\n";
            else if (global_mem_cache_type == CL_READ_ONLY_CACHE)
                params.err += "Global memory cache type: CL_READ_ONLY_CACHE\n";
            else if (global_mem_cache_type == CL_READ_WRITE_CACHE)
                params.err += "Global memory cache type: CL_READ_WRITE_CACHE\n";
            params.err += "Global memory cache size: " + std::to_string(device.get_info<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>() / 1024) + " KB\n";
            params.err += "Global memory size: " + std::to_string(device.get_info<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024 * 1024)) + " MB\n";
            params.err += "Max constant buffer size: " + std::to_string(device.get_info<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>() / 1024) + " KB\n";
            params.err += "Max constant arguments: " + std::to_string(device.get_info<CL_DEVICE_MAX_CONSTANT_ARGS>()) + "\n";
            params.err += "Local memory type: " + std::string{ device.get_info<CL_DEVICE_LOCAL_MEM_TYPE>() == CL_LOCAL ? "CL_LOCAL" : "CL_GLOBAL" } + "\n";
            params.err += "Local memory size: " + std::to_string(device.get_info<CL_DEVICE_LOCAL_MEM_SIZE>() / 1024) + " KB\n";
            params.err += "Available: " + std::string{ device.get_info<CL_DEVICE_AVAILABLE>() ? "CL_TRUE" : "CL_FALSE" } + "\n";
            params.err += "Compiler available: " + std::string{ device.get_info<CL_DEVICE_COMPILER_AVAILABLE>() ? "CL_TRUE" : "CL_FALSE" } + "\n";
            params.err += "OpenCL C version: " + device.get_info<CL_DEVICE_OPENCL_C_VERSION>() + "\n";
            params.err += "Linker available: " + std::string{ device.get_info<CL_DEVICE_LINKER_AVAILABLE>() ? "CL_TRUE" : "CL_FALSE" } + "\n";
            params.err += "Image max buffer size: " + std::to_string(device.get_info<size_t>(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE) / 1024) + " KB" + "\n";
            params.err += "Out of order (on host): " + std::string{ !!(device.get_info<CL_DEVICE_QUEUE_ON_HOST_PROPERTIES>() & 1) ? "CL_TRUE" : "CL_FALSE" } + "\n";
            params.err += "Out of order (on device): " + std::string{ !!(device.get_info<CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES>() & 1) ? "CL_TRUE" : "CL_FALSE" };

            //AVS_Value cl{ avs_new_value_clip(clip) };
            //AVS_Value args_[2]{ cl, avs_new_value_string(err.c_str()) };
            //v = env->Invoke("Text", avs_new_value_array(args_, 2), 0);

            //avs_release_value(cl);
            //avs_release_clip(clip);

            //return v;
        }

        if (field == -2 || field > 1)
        {
            if (vi.num_frames > INT_MAX / 2)
                throw "resulting clip is too long";

            vi.num_frames <<= 1;

            int64_t fps_n{ vi.fps_numerator };
            int64_t fps_d{ vi.fps_denominator };
            muldivRational(&fps_n, &fps_d, 2, 1);
            vi.fps_numerator = static_cast<unsigned>(fps_n);
            vi.fps_denominator = static_cast<unsigned>(fps_d);
        }

        if (dh)
            vi.height <<= 1;

        if (dw)
            vi.width <<= 1;

        const int peak{ (1 << vi.BitsPerComponent()) - 1 };

        std::string weightsPath{ boost::dll::this_line_location().parent_path().generic_string() + "/nnedi3_weights.bin" };

        FILE* weightsFile{ nullptr };
#ifdef _WIN32
        const int requiredSize{ MultiByteToWideChar(CP_UTF8, 0, weightsPath.c_str(), -1, nullptr, 0) };
        std::unique_ptr<wchar_t[]> wbuffer{ std::make_unique<wchar_t[]>(requiredSize) };
        MultiByteToWideChar(CP_UTF8, 0, weightsPath.c_str(), -1, wbuffer.get(), requiredSize);
        weightsFile = _wfopen(wbuffer.get(), L"rb");
#else
        weightsFile = std::fopen(weightsPath.c_str(), "rb");
#endif

#if !defined(_WIN32) && defined(NNEDI3_DATADIR)
        if (!weightsFile)
        {
            weightsPath = std::string{ NNEDI3_DATADIR } + "/nnedi3_weights.bin";
            weightsFile = std::fopen(weightsPath.c_str(), "rb");
        }
#endif
        if (!weightsFile)
            throw "error opening file " + weightsPath + " (" + std::strerror(errno) + ")";

        if (std::fseek(weightsFile, 0, SEEK_END))
        {
            std::fclose(weightsFile);
            throw "error seeking to the end of file " + weightsPath + " (" + std::strerror(errno) + ")";
        }

        constexpr long correctSize{ 13574928 }; // Version 0.9.4 of the Avisynth plugin
        const long weightsSize{ std::ftell(weightsFile) };

        if (weightsSize == -1)
        {
            std::fclose(weightsFile);
            throw "error determining the size of file " + weightsPath + " (" + std::strerror(errno) + ")";
        }
        else if (weightsSize != correctSize)
        {
            std::fclose(weightsFile);
            throw "incorrect size of file " + weightsPath + ". Should be " + std::to_string(correctSize) + " bytes, but got " + std::to_string(weightsSize) + " bytes instead";
        }

        std::rewind(weightsFile);

        float* bdata{ reinterpret_cast<float*>(malloc(correctSize)) };
        const size_t bytesRead{ std::fread(bdata, 1, correctSize, weightsFile) };

        if (bytesRead != correctSize)
        {
            std::fclose(weightsFile);
            free(bdata);
            throw "error reading file " + weightsPath + ". Should read " + std::to_string(correctSize) + " bytes, but read " + std::to_string(bytesRead) + " bytes instead";
        }

        std::fclose(weightsFile);

        constexpr int dims0{ 49 * 4 + 5 * 4 + 9 * 4 };
        constexpr int dims0new{ 4 * 65 + 4 * 5 };
        const int dims1{ nnsTable[nns] * 2 * (xdiaTable[nsize] * ydiaTable[nsize] + 1) };
        int dims1tsize{ 0 };
        int dims1offset{ 0 };

        for (int j{ 0 }; j < numNNS; ++j)
        {
            for (int i{ 0 }; i < numNSIZE; ++i)
            {
                if (i == nsize && j == nns)
                    dims1offset = dims1tsize;

                dims1tsize += nnsTable[j] * 2 * (xdiaTable[i] * ydiaTable[i] + 1) * 2;
            }
        }

        float* weights0{ new float[std::max(dims0, dims0new)] };
        float* weights1{ new float[dims1 * 2] };

        // Adjust prescreener weights
        if (pscrn == 2) // using new prescreener
        {
            int* offt{ reinterpret_cast<int*>(calloc(4 * 64, sizeof(int))) };

            for (int j{ 0 }; j < 4; ++j)
            {
                for (int k{ 0 }; k < 64; ++k)
                    offt[j * 64 + k] = ((k >> 3) << 5) + ((j & 3) << 3) + (k & 7);
            }

            const float* bdw{ bdata + dims0 + dims0new * (pscrn - 2) };
            short* ws{ reinterpret_cast<short*>(weights0) };
            float* wf{ reinterpret_cast<float*>(&ws[4 * 64]) };
            double mean[4]{ 0.0, 0.0, 0.0, 0.0 };

            // Calculate mean weight of each first layer neuron
            for (int j{ 0 }; j < 4; ++j)
            {
                double cmean{ 0.0 };

                for (int k{ 0 }; k < 64; ++k)
                    cmean += bdw[offt[j * 64 + k]];

                mean[j] = cmean / 64.0;
            }

            const double half{ peak / 2.0 };

            // Factor mean removal and 1.0/half scaling into first layer weights. scale to int16 range
            for (int j{ 0 }; j < 4; ++j)
            {
                double mval{ 0.0 };
                for (int k{ 0 }; k < 64; ++k)
                    mval = std::max(mval, std::abs((bdw[offt[j * 64 + k]] - mean[j]) / half));

                const double scale{ 32767.0 / mval };

                for (int k{ 0 }; k < 64; ++k)
                    ws[offt[j * 64 + k]] = roundds(((bdw[offt[j * 64 + k]] - mean[j]) / half) * scale);

                wf[j] = static_cast<float>(mval / 32767.0);
            }

            memcpy(wf + 4, bdw + 4 * 64, (dims0new - 4 * 64) * sizeof(float));
            free(offt);
        }
        else // using old prescreener
        {
            double mean[4]{ 0.0, 0.0, 0.0, 0.0 };

            // Calculate mean weight of each first layer neuron
            for (int j{ 0 }; j < 4; ++j)
            {
                double cmean{ 0.0 };

                for (int k{ 0 }; k < 48; ++k)
                    cmean += bdata[j * 48 + k];

                mean[j] = cmean / 48.0;
            }

            const double half{ ((vi.ComponentSize() < 4) ? peak : 1.0) / 2.0 };

            // Factor mean removal and 1.0/half scaling into first layer weights
            for (int j{ 0 }; j < 4; ++j)
            {
                for (int k{ 0 }; k < 48; ++k)
                    weights0[j * 48 + k] = static_cast<float>((bdata[j * 48 + k] - mean[j]) / half);
            }

            memcpy(weights0 + 4 * 48, bdata + 4 * 48, (dims0 - 4 * 48) * sizeof(float));
        }

        // Adjust prediction weights
        for (int i{ 0 }; i < 2; ++i)
        {
            const float* bdataT{ bdata + dims0 + dims0new * 3 + dims1tsize * etype + dims1offset + i * dims1 };
            float* weightsT{ weights1 + i * dims1 };
            const int nnst{ nnsTable[nns] };
            const int asize{ xdiaTable[nsize] * ydiaTable[nsize] };
            const int boff{ nnst * 2 * asize };
            double* mean{ reinterpret_cast<double*>(calloc(asize + 1 + nnst * 2, sizeof(double))) };

            // Calculate mean weight of each neuron (ignore bias)
            for (int j{ 0 }; j < nnst * 2; ++j)
            {
                double cmean{ 0.0 };

                for (int k{ 0 }; k < asize; ++k)
                    cmean += bdataT[j * asize + k];

                mean[asize + 1 + j] = cmean / asize;
            }

            // Calculate mean softmax neuron
            for (int j{ 0 }; j < nnst; ++j)
            {
                for (int k{ 0 }; k < asize; ++k)
                    mean[k] += bdataT[j * asize + k] - mean[asize + 1 + j];

                mean[asize] += bdataT[boff + j];
            }

            for (int j{ 0 }; j < asize + 1; ++j)
                mean[j] /= nnst;

            // Factor mean removal into weights, and remove global offset from softmax neurons
            for (int j{ 0 }; j < nnst * 2; ++j)
            {
                for (int k{ 0 }; k < asize; ++k)
                {
                    const double q{ (j < nnst) ? mean[k] : 0.0 };
                    weightsT[j * asize + k] = static_cast<float>(bdataT[j * asize + k] - mean[asize + 1 + j] - q);
                }

                weightsT[boff + j] = static_cast<float>(bdataT[boff + j] - (j < nnst ? mean[asize] : 0.0));
            }

            free(mean);
        }

        free(bdata);

        const int xdia{ xdiaTable[nsize] };
        const int ydia{ ydiaTable[nsize] };
        const int asize{ xdiaTable[nsize] * ydiaTable[nsize] };
        const int xdiad2m1{ std::max(xdia, (pscrn == 1) ? 12 : 16) / 2 - 1 };
        const int ydiad2m1{ ydia / 2 - 1 };
        const int xOffset{ (xdia == 8) ? (pscrn == 1 ? 2 : 4) : 0 };
        const int inputWidth{ std::max(xdia, (pscrn == 1) ? 12 : 16) + 32 - 1 };
        const int inputHeight{ ydia + 16 - 1 };
        const float scaleAsize{ 1.0f / asize };
        const float scaleQual{ 1.0f / qual };

        params.weights0 = boost::compute::buffer{ context, std::max(dims0, dims0new) * sizeof(cl_float), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, weights0 };
        params.weights1Buffer = boost::compute::buffer{ context, dims1 * 2 * sizeof(cl_float), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, weights1 };
        delete[] weights0;
        delete[] weights1;

        if (static_cast<size_t>(dims1 * 2) > device.get_info<size_t>(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE))
            throw "the device's image max buffer size is too small. Reduce nsize/nns...or buy a new graphics card";

        boost::compute::program program;
        try
        {
            std::ostringstream options;
            options.imbue(std::locale{ "C" });
            options.precision(16);
            options.setf(std::ios::fixed, std::ios::floatfield);
            options << "-cl-denorms-are-zero -cl-fast-relaxed-math -Werror";
            options << " -D QUAL=" << qual;
            if (pscrn == 1)
            {
                options << " -D PRESCREEN=prescreenOld";
                options << " -D USE_OLD_PSCRN=1";
                options << " -D USE_NEW_PSCRN=0";
            }
            else
            {
                options << " -D PRESCREEN=prescreenNew";
                options << " -D USE_OLD_PSCRN=0";
                options << " -D USE_NEW_PSCRN=1";
            }
            options << " -D PSCRN_OFFSET=" << (pscrn == 1 ? 5 : 6);
            options << " -D DIMS1=" << dims1;
            options << " -D NNS=" << nnsTable[nns];
            options << " -D NNS2=" << (nnsTable[nns] * 2);
            options << " -D XDIA=" << xdia;
            options << " -D YDIA=" << ydia;
            options << " -D ASIZE=" << asize;
            options << " -D XDIAD2M1=" << xdiad2m1;
            options << " -D YDIAD2M1=" << ydiad2m1;
            options << " -D X_OFFSET=" << xOffset;
            options << " -D INPUT_WIDTH=" << inputWidth;
            options << " -D INPUT_HEIGHT=" << inputHeight;
            options << " -D SCALE_ASIZE=" << scaleAsize << "f";
            options << " -D SCALE_QUAL=" << scaleQual << "f";
            options << " -D PEAK=" << peak;
            if (!(dh || dw))
            {
                options << " -D Y_OFFSET=" << (ydia - 1);
                options << " -D Y_STEP=2";
                options << " -D Y_STRIDE=32";
            }
            else
            {
                options << " -D Y_OFFSET=" << (ydia / 2);
                options << " -D Y_STEP=1";
                options << " -D Y_STRIDE=16";
            }

            program = boost::compute::program::build_with_source(source, context, options.str());
        }
        catch (const boost::compute::opencl_error& error)
        {
            throw error.error_string() + "\n" + program.build_log();
        }

        if (vi.ComponentSize() < 4)
            kernel = program.create_kernel("filter_uint");
        else
            kernel = program.create_kernel("filter_float");

        //const int st{ avs_defined(avs_array_elt(args, St)) ? avs_as_bool(avs_array_elt(args, St)) : !!(device.get_info<CL_DEVICE_QUEUE_ON_HOST_PROPERTIES>() & 1) };
        cl_image_format imageFormat;

        switch (vi.ComponentSize())
        {
            case 1:
            {
                imageFormat = { CL_R, CL_UNSIGNED_INT8 };
                filter = (st) ? &Nnedi3CL::resize_plane_c<uint8_t, true> : &Nnedi3CL::resize_plane_c<uint8_t, false>;
                break;
            }
            case 2:
            {
                imageFormat = { CL_R, CL_UNSIGNED_INT16 };
                filter = (st) ? &Nnedi3CL::resize_plane_c<uint16_t, true> : &Nnedi3CL::resize_plane_c<uint16_t, false>;
                break;
            }
            default:
            {
                imageFormat = { CL_R, CL_FLOAT };
                filter = (st) ? &Nnedi3CL::resize_plane_c<float, true> : &Nnedi3CL::resize_plane_c<float, false>;
            }
        }

        src = boost::compute::image2d{ context,
                                   static_cast<size_t>(vi.width),
                                   static_cast<size_t>(vi.height),
                                   boost::compute::image_format{ imageFormat },
                                   CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY };

        dst = boost::compute::image2d{ context,
                                   static_cast<size_t>(std::max(vi.width, vi.height)),
                                   static_cast<size_t>(std::max(vi.width, vi.height)),
                                   boost::compute::image_format{ imageFormat },
                                   CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY };

        tmp = (dh && dw) ? boost::compute::image2d{ context,
                                                      static_cast<size_t>(std::max(vi.width, vi.height)),
                                                      static_cast<size_t>(std::max(vi.width, vi.height)),
                                                      boost::compute::image_format{ imageFormat },
                                                      CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS }
        : boost::compute::image2d{};

        {
            constexpr cl_image_format format{ CL_R, CL_FLOAT };

            cl_image_desc desc;
            desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            desc.image_width = dims1 * 2;
            desc.image_height = 1;
            desc.image_depth = 1;
            desc.image_array_size = 0;
            desc.image_row_pitch = 0;
            desc.image_slice_pitch = 0;
            desc.num_mip_levels = 0;
            desc.num_samples = 0;
#ifdef BOOST_COMPUTE_CL_VERSION_2_0
            desc.mem_object = weights1Buffer.get();
#else
            desc.buffer = d->weights1Buffer.get();
#endif

            cl_int error{ 0 };

            cl_mem mem{ clCreateImage(context, 0, &format, &desc, nullptr, &error) };
            if (!mem)
                BOOST_THROW_EXCEPTION(boost::compute::opencl_error(error));

            params.weights1 = mem;
        }
    }
    catch (const std::string& error)
    {
        params.err = "NNEDI3CL: " + error;
        //v = avs_new_value_error(err.c_str());
    }
    catch (const boost::compute::no_device_found& error)
    {
        params.err = std::string{ "NNEDI3CL: " } + error.what();
        //v = avs_new_value_error(err.c_str());
    }
    catch (const boost::compute::opencl_error& error)
    {
        params.err = "NNEDI3CL: " + error.error_string();
        //v = avs_new_value_error(err.c_str());
    }

    /*if (!avs_defined(v))
    {
        v = avs_new_value_clip(clip);

        fi->user_data = reinterpret_cast<void*>(params);
        fi->get_frame = NNEDI3CL_get_frame;
        fi->set_cache_hints = NNEDI3CL_set_cache_hints;
        fi->free_filter = free_NNEDI3CL;
    }

    avs_release_clip(clip);

    return v;*/
}

//Nnedi3CL::~Nnedi3CL()
//{
    /*for (int i = 0; i < static_cast<int>(out1.size()); ++i)
    {
        delete_coeff_table(out1[i]);
        delete out1[i];
    }

    delete[] init_lut->lut;
    delete init_lut;*/
//}

/*PVideoFrame Nnedi3CL::GetFrame(int n, IScriptEnvironment* env)
{
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrameP(vi, &src);

    (this->*filter)(src, dst, env);

    if (vi.Is420() || vi.Is422() || vi.IsYV411())
    {
        if (cplace == "mpeg2")
            env->propSetInt(env->getFramePropsRW(dst), "_ChromaLocation", 0, 0);
        else if (cplace == "mpeg1")
            env->propSetInt(env->getFramePropsRW(dst), "_ChromaLocation", 1, 0);
        else
            env->propSetInt(env->getFramePropsRW(dst), "_ChromaLocation", 2, 0);
    }

    return dst;
}

static AVS_VideoFrame* AVSC_CC JincResize_GetFrame(AVS_FilterInfo* fi, int n)
{
    JincResize* d = reinterpret_cast<JincResize*>(fi->user_data);
    AVS_ScriptEnvironment* env = fi->env;
    AVS_VideoInfo* vi = &fi->vi;

    AVS_VideoFrame* src = avs_get_frame(fi->child, n);
    if (!src)
        return nullptr;

    AVS_VideoFrame* dst = avs_new_video_frame_p(env, vi, src);

    (d->*d->process_frame)(src, dst, vi);

    if ((avs_is_420(vi) || avs_is_422(vi) || avs_is_yv411(vi)))
    {
        if (d->cplace == "mpeg2")
            avs_prop_set_int(env, avs_get_frame_props_rw(env, dst), "_ChromaLocation", 0, 0);
        else if (d->cplace == "mpeg1")
            avs_prop_set_int(env, avs_get_frame_props_rw(env, dst), "_ChromaLocation", 1, 0);
        else
            avs_prop_set_int(env, avs_get_frame_props_rw(env, dst), "_ChromaLocation", 2, 0);
    }

    avs_release_video_frame(src);

    return dst;
}*/

/*PVideoFrame Nnedi3CL::GetFrame(int n, IScriptEnvironment* env)
{
    //NNEDI3CLData* d{ static_cast<NNEDI3CLData*>(fi->user_data) };
    const NNEDI3CLData* const __restrict d;

    const int field_no_prop = [&]()
    {
        if (d->field == -1)
            return avs_get_parity(fi->child, n) ? 1 : 0;
        else if (d->field == -2)
            return avs_get_parity(fi->child, n >> 1) ? 3 : 2;
        else
            return -1;
    }();

    int field{ (d->field > -1) ? d->field : field_no_prop };

    PVideoFrame src = child->GetFrame((field > 1) ? (n >> 1) : n, env);

    if (!src)
        return nullptr;

    PVideoFrame dst = env->NewVideoFrameP(vi, &src);

    if (d->field < 0)
    {
        int err;
        const int64_t field_based = env->propGetInt("_FieldBased", 0, &err);
        if (err == 0)
        {
            if (field_based == 1)
                field = 0;
            else if (field_based == 2)
                field = 1;

            if (d->field > 1 || field_no_prop > 1)
            {
                if (field_based == 0)
                    field -= 2;

                field = static_cast<int>((n & 1) ? (field == 0) : (field == 1));
            }
        }
        else
        {
            if (field > 1)
            {
                field -= 2;
                field = static_cast<int>((n & 1) ? (field == 0) : (field == 1));
            }
        }
    }
    else
    {
        if (field > 1)
        {
            field -= 2;
            field = static_cast<int>((n & 1) ? (field == 0) : (field == 1));
        }
    }

    try
    {
        this->*filter(&src, &dst, field, d);
    }
    catch (const boost::compute::opencl_error& error)
    {
        d->err = "NNEDI3CL: " + error.error_string();
        //fi->error = d->err.c_str();
        //avs_release_video_frame(src);
        //avs_release_video_frame(dst);

        return nullptr;
    }

    AVS_Map* props{ env->getFramePropsRW(dst) };

    env->propGetInt(props, "_FieldBased", 0, 0);

    if (d->field > 1 || field_no_prop > 1)
    {
        int errNum;
        int errDen;
        int64_t durationNum{ env->propGetInt(props, "_DurationNum", 0, &errNum) };
        int64_t durationDen{ env->propGetInt(props, "_DurationDen", 0, &errDen) };
        if (errNum == 0 && errDen == 0)
        {
            muldivRational(&durationNum, &durationDen, 1, 2);
            env->propGetInt("_DurationNum", durationNum, 0);
            env->propGetInt("_DurationDen", durationDen, 0);
        }
    }

    //avs_release_video_frame(src);

    return dst;
}*/

AVSValue __cdecl Create_NNEDI3CL(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    const VideoInfo& vi = args[0].AsClip()->GetVideoInfo();

    return new Nnedi3CL(
        args[0].AsClip(),
        args[1].AsInt(-1),
        args[2].AsBool(false),
        args[3].AsBool(false),
        args[4].AsInt(0),
        args[5].AsInt(6),
        args[6].AsInt(1),
        args[7].AsInt(1),
        args[8].AsInt(0),
        args[9].AsInt((vi.NumComponents() < 4) ? 2 : 1),
        args[10].AsInt(-1),
        args[11].AsBool(false),
        args[12].AsBool(false),
        args[13].AsBool(false),
        args[14].AsBool(false),
        env);
}

/*const char* AVSC_CC avisynth_c_plugin_init(AVS_ScriptEnvironment* env)
{
    avs_add_function(env, "NNEDI3CL", "c[field]i[dh]b[dw]b[planes]i[nsize]i[nns]i[qual]i[etype]i[pscrn]i[device]i[list_device]b[info]b[st]b[luma]b", Create_NNEDI3CL, 0);
    return "NNEDI3CL";
}*/
