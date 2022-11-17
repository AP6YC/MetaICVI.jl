using Revise


module TestSaveLoad
    using PyCall, JLD, PyCallJLD, ScikitLearn
    # @sk_import linear_model: SGDClassifier
    struct MyStruct
        classifier::PyCall.PyObject
    end

    function MyStruct(new_classifier::Bool)
        # @sk_import linear_model: SGDClassifier
        @eval begin
            if !@isdefined SGDClassifier
                @sk_import linear_model: SGDClassifier
            end
        end
        if new_classifier
            mystruct = MyStruct(SGDClassifier())
        else
            mystruct = MyStruct(load_classifier())
        end
        return mystruct
    end

    function save_classifier(my_struct::MyStruct)
        JLD.save("mystruct.jld", "classifier", my_struct.classifier)
        return
    end

    function load_classifier()
        # return JLD.load("mystruct.jld", "classifier")
        return @eval JLD.load("mystruct.jld", "classifier")
    end

    export MyStruct, save_classifier, load_classifier
end


using .TestSaveLoad

ms = MyStruct(true)

save_classifier(ms)

msnew = MyStruct(false)
