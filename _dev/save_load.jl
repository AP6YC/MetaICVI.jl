using Revise
using JLD2
using ScikitLearn

using MetaICVI

using PyCall

const dumps = PyNULL()
const loads = PyNULL()

function __init__()
    pickle = pyimport(PyCall.pyversion.major ≥ 3 ? "pickle" : "cPickle")
    copy!(dumps, pickle.dumps)
    copy!(loads, pickle.loads)
end

struct PyObjectSerialization
    repr::Vector{UInt8}
end

# function JLD.writeas(pyo::PyObject)
function get_saver(pyo::PyObject)
    b = PyCall.PyBuffer(pycall(dumps, PyObject, pyo))
    # We need a `copy` here because the PyBuffer might be GC'ed after we've
    # left this scope, but see
    # https://github.com/JuliaPy/PyCallJLD.jl/pull/3/files/17b052d018f79905baf855b40e440d2cacc171ae#r115525173
    PyObjectSerialization(copy(unsafe_wrap(Array, Ptr{UInt8}(pointer(b)), sizeof(b))))
end

# jldopen(filename, "w") do file
#     file[SINGLE_OBJECT_NAME] = x
#   end

# function JLD.readas(pyo_ser::PyObjectSerialization)
function get_loader(pyo_ser::PyObjectSerialization)
    pycall(loads, PyObject,
           PyObject(PyCall.@pycheckn ccall(@pysym(PyCall.PyString_FromStringAndSize),
                                           PyPtr, (Ptr{UInt8}, Int),
                                           pyo_ser.repr, sizeof(pyo_ser.repr))))
end

@info "--- CREATING FIRST MODULE ---"
# Create the module
opts = MetaICVIOpts(
    # fail_on_missing = true
    fail_on_missing = false
)
metaicvi = MetaICVIModule(opts)

# 10.5281/zenodo.7327501
# const MetaICVIClassifier = ScikitLearn.Skcore.FitBit
# const MetaICVIClassifier = PyCall.PyObject
# MetaICVIClassifier = PyCall.PyObject
# if !@isdefined SGDClassifier
#     @sk_import linear_model: SGDClassifier
# end
# MetaICVIClassifier = SGDClassifier

@info "--- CREATING SECOND MODULE ---"
# Create the module
opts = MetaICVIOpts(
    fail_on_missing = true
)

new_metaicvi = MetaICVIModule(opts)

# __init__()

# a = MetaICVIClassifier(load_object("../data/models/classifier.jld2"))
# sizeof(a)
# a = get_saver(metaicvi.classifier)
# b = PyCall.PyBuffer(pycall(dumps, PyObject, metaicvi.classifier))