# MetaICVI

A Julia implementation of the Meta-ICVI method as a separate package.

| **Documentation**  | **Build Status** | **Coverage** |
|:------------------:|:----------------:|:------------:|
| [![Stable][docs-stable-img]][docs-stable-url] [![Dev][docs-dev-img]][docs-dev-url] | [![Build Status][ci-img]][ci-url] | [![Codecov][codecov-img]][codecov-url]

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://AP6YC.github.io/MetaICVI.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://AP6YC.github.io/MetaICVI.jl/dev

[ci-img]: https://github.com/AP6YC/MetaICVI.jl/workflows/CI/badge.svg
[ci-url]: https://github.com/AP6YC/MetaICVI.jl/actions

[codecov-img]: https://codecov.io/gh/AP6YC/MetaICVI.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/AP6YC/MetaICVI.jl

[issues-url]: https://github.com/AP6YC/MetaICVI.jl/issues

## Table of Contents

- [MetaICVI](#metaicvi)
  - [Table of Contents](#table-of-contents)
  - [Usage](#usage)
    - [Basic Usage](#basic-usage)
    - [Advanced Usage](#advanced-usage)
  - [Contributing](#contributing)
  - [Credits](#credits)
    - [Authors](#authors)
    - [License](#license)

## Usage

### Basic Usage

Create a MetaICI module with the default constructor

```julia
    metaicvi = MetaICVIModule()
```

and retrieve the MetaICVI value iteratively with

```julia
    get_metaicvi(metaicvi, sample, label)
```

where `sample` is a real-valued vector and `label` is an integer.

### Advanced Usage

You can specify the MetaICVI options with

```julia
    opts = MetaICVIOpts(
        icvi_window = 5,
        correlation_window = 5,
        n_rocket = 5,
        rocket_file = "data/models/rocket.jld2",
        classifier_file = "data/models/classifier.jld",
        display = true,
        fail_on_missing = false
    )
    metaicvi = MetaICVIModule(opts)
```

The options are

- `icvi_window`: the number of ICVI criterion values to compute rank correlation across.
- `correlation_window`: the number of correlations to compute rocket features across.
- `rocket_file`: filename of a saved RocketModule.
- `classifier_file`: filename of a saved linear classifier.
- `display`: boolean flag for logging info.
- `fail_on_missing`: boolean flag for crashing if missing rocket and/or classifier files.

## Contributing

Please raise an [issue][issues-url].

## Credits

### Authors

- Sasha Petrenko <sap625@mst.edu>

### License

This software is developed by the Applied Computational Intelligence Laboratory (ACIL) of the Missouri University of Science and Technology (S&amp;T) under the supervision of Teledyne Technologies for the DARPA L2M program.
Read the [License](LICENSE).
