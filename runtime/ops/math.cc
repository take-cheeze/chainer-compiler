#include <chainerx/routines/activation.h>
#include <chainerx/routines/arithmetic.h>
#include <chainerx/routines/connection.h>
#include <chainerx/routines/creation.h>
#include <chainerx/routines/explog.h>
#include <chainerx/routines/hyperbolic.h>
#include <chainerx/routines/linalg.h>
#include <chainerx/routines/logic.h>
#include <chainerx/routines/manipulation.h>
#include <chainerx/routines/misc.h>
#include <chainerx/routines/rounding.h>
#include <chainerx/routines/trigonometric.h>

#include <common/log.h>
#include <runtime/chainerx_util.h>
#include <runtime/gen_chxvm_ops.h>

#include <numeric>

#if CHAINER_COMPILER_ENABLE_CLBLAS
#include <clBLAS.h>
#endif

namespace chainer_compiler {
namespace runtime {

namespace {

chainerx::Array Pow(chainerx::Array a, chainerx::Array b) {
    return chainerx::Power(a, b);
}

// TODO(hamaji): Implement type coersion in ChainerX.
chainerx::Dtype CoerceDtype(chainerx::Dtype dtype0, chainerx::Dtype dtype1) {
    if (dtype0 == dtype1) return dtype0;
    if (IsFloat(dtype0) && !IsFloat(dtype1)) return dtype0;
    if (!IsFloat(dtype0) && IsFloat(dtype1)) return dtype1;
    if (chainerx::GetItemSize(dtype0) > chainerx::GetItemSize(dtype1)) return dtype0;
    if (chainerx::GetItemSize(dtype0) < chainerx::GetItemSize(dtype1)) return dtype1;
    if (dtype1 == chainerx::Dtype::kBool) return dtype0;
    if (dtype0 == chainerx::Dtype::kBool) return dtype1;
    if (dtype0 == chainerx::Dtype::kUInt8 || dtype1 == chainerx::Dtype::kUInt8) return chainerx::Dtype::kInt16;
    CHECK(false) << "Unknown type coerce: " << dtype0 << " vs " << dtype1;
}

std::tuple<chainerx::Array, chainerx::Array> CoerceBinary(const chainerx::Array& a, const chainerx::Array& b) {
    chainerx::Array ax = a;
    chainerx::Array bx = b;
    chainerx::Dtype dtype = CoerceDtype(a.dtype(), b.dtype());
    ax = CastTo(ax, dtype);
    bx = CastTo(bx, dtype);
    return std::tie(ax, bx);
}

}  // namespace

chainerx::Array AddOp::RunImpl(ChxVMState* st, const chainerx::Array& a, const chainerx::Array& b) {
    auto t = CoerceBinary(a, b);
    return std::get<0>(t) + std::get<1>(t);
}

chainerx::Array SubOp::RunImpl(ChxVMState* st, const chainerx::Array& a, const chainerx::Array& b) {
    auto t = CoerceBinary(a, b);
    return std::get<0>(t) - std::get<1>(t);
}

chainerx::Array MulOp::RunImpl(ChxVMState* st, const chainerx::Array& a, const chainerx::Array& b) {
    auto t = CoerceBinary(a, b);
    return std::get<0>(t) * std::get<1>(t);
}

chainerx::Array DivOp::RunImpl(ChxVMState* st, const chainerx::Array& a0, const chainerx::Array& b0) {
    chainerx::Array a, b;
    std::tie(a, b) = CoerceBinary(a0, b0);
    // TODO(hamaji): Come up with a better idea to handle cross device ops.
    if (&a.device() != &b.device() && b.GetTotalSize() == 1) {
        if (IsFloat(a.dtype())) {
            return chainerx::TrueDivide(a, chainerx::AsScalar(b));
        } else {
            return chainerx::FloorDivide(a, chainerx::AsScalar(b));
        }
    }
    if (IsFloat(a.dtype())) {
        return chainerx::TrueDivide(a, b);
    } else {
        return chainerx::FloorDivide(a, b);
    }
}

chainerx::Array PowOp::RunImpl(ChxVMState* st, const chainerx::Array& a, const chainerx::Array& b) {
    auto t = CoerceBinary(a, b);
    return Pow(std::get<0>(t), std::get<1>(t));
}

chainerx::Array NegOp::RunImpl(ChxVMState* st, const chainerx::Array& a) {
    return -a;
}

chainerx::Array IsNaNOp::RunImpl(ChxVMState* st, const chainerx::Array& a) {
    return chainerx::IsNan(a);
}

chainerx::Array IsInfOp::RunImpl(ChxVMState* st, const chainerx::Array& a) {
    chainerx::Array ret = chainerx::IsInf(a);
    if (detect_negative && detect_positive) {
        return ret;
    }

    const chainerx::Array zeros = chainerx::Zeros(ret.shape(), a.dtype(), a.device());
    if (detect_negative) {
        ret = LogicalAnd(ret, a < zeros);
    }
    if (detect_positive) {
        ret = LogicalAnd(ret, a > zeros);
    }
    return ret;
}

#define DEFINE_UNARY_OP(op)                                                     \
    chainerx::Array op##Op::RunImpl(ChxVMState* st, const chainerx::Array& a) { \
        return chainerx::op(a);                                                 \
    }

#define DEFINE_UNARY_OP_TODO(op)                                                \
    chainerx::Array op##Op::RunImpl(ChxVMState* st, const chainerx::Array& a) { \
        CHECK(false) << "TODO(hamaji): " #op " op not implemented";             \
    }

DEFINE_UNARY_OP(Exp);
DEFINE_UNARY_OP(Log);
DEFINE_UNARY_OP(Sqrt);
DEFINE_UNARY_OP(Reciprocal);
DEFINE_UNARY_OP(Sin);
DEFINE_UNARY_OP(Cos);
DEFINE_UNARY_OP(Tan);
DEFINE_UNARY_OP(Arcsin);
DEFINE_UNARY_OP(Arccos);
DEFINE_UNARY_OP(Arctan);
DEFINE_UNARY_OP(Sinh);
DEFINE_UNARY_OP(Cosh);
DEFINE_UNARY_OP(Arcsinh);
DEFINE_UNARY_OP(Arccosh);
DEFINE_UNARY_OP_TODO(Arctanh);
DEFINE_UNARY_OP(Erf);

chainerx::Array AbsOp::RunImpl(ChxVMState* st, const chainerx::Array& x) {
    return chainerx::Absolute(x);
}

chainerx::Array FloorOp::RunImpl(ChxVMState* st, const chainerx::Array& x) {
    if (!IsFloat(x.dtype())) {
        return x;
    }
    return chainerx::Floor(x);
}

chainerx::Array CeilOp::RunImpl(ChxVMState* st, const chainerx::Array& x) {
    if (!IsFloat(x.dtype())) {
        return x;
    }
    return chainerx::Ceil(x);
}

chainerx::Array ClipOp::RunImpl(ChxVMState* st, const chainerx::Array& x) {
    return chainerx::Minimum(chainerx::Maximum(x, min), max);
}

chainerx::Array MatMulOp::RunImpl(ChxVMState* st, const chainerx::Array& a, const chainerx::Array& b) {
    return NumpyMatMul(a, b);
}

chainerx::Array GemmOp::RunImpl(ChxVMState* st, const chainerx::Array& a, const chainerx::Array& b, const chainerx::Array& c) {
    auto fallback_gemm = [&]() {
        if (alpha == 1.0 && beta == 1.0 && !trans_a && trans_b && c.ndim() == 1) {
            return Linear(a, b, c);
        }

        chainerx::Array xa = a;
        chainerx::Array xb = b;
        if (trans_a) xa = chainerx::Transpose(xa);
        if (trans_b) xb = chainerx::Transpose(xb);
        chainerx::Array r = chainerx::Dot(xa, xb);
        if (alpha != 1.0) r *= alpha;
        if (beta == 0.0) return r;
        chainerx::Array xc = c;
        if (beta != 1.0) xc = xc * beta;
        return r + xc;
    };
#if CHAINER_COMPILER_ENABLE_CLBLAS
    if (a.dtype() != chainerx::Dtype::kFloat32 || b.dtype() != chainerx::Dtype::kFloat32 || c.dtype() != chainerx::Dtype::kFloat32) {
        return fallback_gemm();
    }

    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, 0, 0};
    cl_context ctx = 0;
    cl_command_queue queue = 0;
    cl_mem bufA, bufB, bufC;
    cl_event event = NULL;

    chainerx::Array cont_a = chainerx::AsContiguous(a);
    chainerx::Array cont_b = chainerx::AsContiguous(b);
    chainerx::Array cont_c = chainerx::AsContiguous(c);

    /* Setup OpenCL environment. */
    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(ctx, device, 0, &err);

    /* Setup clBLAS */
    err = clblasSetup();

    int64_t m = cont_a.shape()[0], k = cont_a.shape()[1], n = cont_b.shape()[1];

    /* Prepare OpenCL memory objects and place matrices inside them. */
    bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, m * k * sizeof(float), NULL, &err);
    bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, k * n * sizeof(float), NULL, &err);
    bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, m * n * sizeof(float), NULL, &err);

    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, m * k * sizeof(float), cont_a.raw_data(), 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, k * n * sizeof(float), cont_b.raw_data(), 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, m * n * sizeof(float), cont_c.raw_data(), 0, NULL, NULL);

    /* Call clBLAS extended function. Perform gemm for the lower right sub-matrices */
    err = clblasSgemm(
            clblasRowMajor,
            trans_a ? clblasTrans : clblasNoTrans,
            trans_b ? clblasTrans : clblasNoTrans,
            m,
            n,
            k,
            alpha,
            bufA,
            0,
            cont_a.shape()[0],
            bufB,
            0,
            cont_b.shape()[0],
            beta,
            bufC,
            0,
            cont_c.shape()[0],
            1,
            &queue,
            0,
            NULL,
            &event);

    /* Wait for calculations to be finished. */
    err = clWaitForEvents(1, &event);

    std::vector<float> result(m * n);

    /* Fetch results of calculations from GPU memory. */
    err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, m * n * sizeof(float), result.data(), 0, NULL, NULL);

    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufC);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufA);

    /* Finalize work with clBLAS */
    clblasTeardown();

    /* Release OpenCL working objects. */
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return MakeArray(chainerx::Dtype::kFloat32, {m, n}, result.data());
#else
    return fallback_gemm();
#endif
}

chainerx::Array MaxOp::RunImpl(ChxVMState* st, const std::vector<chainerx::Array>& inputs) {
    CHECK_LT(0, inputs.size());
    chainerx::Array result = inputs[0];
    for (size_t i = 1; i < inputs.size(); ++i) {
        result = Maximum(result, inputs[i]);
    }
    return result;
}

chainerx::Array MinOp::RunImpl(ChxVMState* st, const std::vector<chainerx::Array>& inputs) {
    CHECK_LT(0, inputs.size());
    chainerx::Array result = inputs[0];
    for (size_t i = 1; i < inputs.size(); ++i) {
        result = Minimum(result, inputs[i]);
    }
    return result;
}

chainerx::Array SignOp::RunImpl(ChxVMState* st, chainerx::Array const& input) {
    return chainerx::Sign(input);
}

}  // namespace runtime
}  // namespace chainer_compiler
