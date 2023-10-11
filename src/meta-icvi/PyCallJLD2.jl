module PyCallJLD2

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using PyCall, JLD2

# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

const dumps = PyNULL()
const loads = PyNULL()

# Lock during pycalls
# https://github.com/JuliaPy/PyCall.jl/issues/882
const PYLOCK = Ref{ReentrantLock}()

function __init__()
    pickle = pyimport(PyCall.pyversion.major ≥ 3 ? "pickle" : "cPickle")
    copy!(dumps, pickle.dumps)
    copy!(loads, pickle.loads)
    PYLOCK[] = ReentrantLock()
end

struct PyObjectSerialization
    repr::Vector{UInt8}
end

JLD2.writeas(::Type{PyObject}) = PyObjectSerialization

pylock(f::Function) = Base.lock(f, PYLOCK[])

function JLD2.wconvert(::Type{PyObjectSerialization}, pyo::PyObject)
    # __init__()
    b = PyCall.PyBuffer(pycall(dumps, PyObject, pyo))

    # We need a `copy` here because the PyBuffer might be GC'ed after we've
    # left this scope, but see
    # https://github.com/JuliaPy/PyCallJLD.jl/pull/3/files/17b052d018f79905baf855b40e440d2cacc171ae#r115525173
    serialized = pylock() do
        PyObjectSerialization(
            copy(
                unsafe_wrap(
                    Array,
                    Ptr{UInt8}(pointer(b)),
                    sizeof(b)
                )
            )
        )
    end

    return serialized
end

function JLD2.rconvert(::Type{PyObject}, pyo_ser::PyObjectSerialization)
    reconstructed = pylock() do
        pycall(
            loads,
            PyObject,
            # pyincref(PyObject(
            PyObject(
                PyCall.@pycheckn ccall(
                    @pysym(PyCall.PyString_FromStringAndSize),
                    # @pysym(PyCall.PyUnicode_FromStringAndSize),
                    PyPtr,
                    (Ptr{UInt8}, Int),
                    pyo_ser.repr,
                    sizeof(pyo_ser.repr)
                )
            )
            # ))
        )
    end

    return reconstructed
end

end
