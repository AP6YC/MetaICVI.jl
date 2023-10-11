# ---
# title: MetaICVI
# id: metaicvi-example
# date: 2023-5-15
# cover: ../assets/clustering.png
# author: "[Sasha Petrenko](https://github.com/AP6YC)"
# julia: 1.9
# description: This demo provides a quick example of how use the MetaICVI module.
# ---

# ## Overview

# This example demonstrates the basic usage of the MetaICVI method, including special considerations on how to load the package and how to use modules in the package.

# ## Setup

# First, we load our dependencies

## Multi-line using statements are permitted in Julia to gather all requirements and compile at once
using
    PyCallJLD,  # For loading Python scikit-learn objects
    MetaICVI    # This project

# Next, we will include some utilties used in the testing process for the sake of loading and formatting the IRIS dataset for our purposes
include("../../../../test/test_utils.jl")
